## Efficient k -Sparse Band-Limited Interpolation with Improved Approximation Ratio

## Yang Cao

Wyoming Seminary ycao4@wyomingseminary.org

## Zhao Song

University of California, Berkeley magic.linuxkde@gmail.com

## Xiaoyu Li

University of New South Wales

7.xiaoyu.li@gmail.com

## Chiwun Yang

Sun Yat-sen University christiannyang37@gmail.com

## Abstract

We consider the task of interpolating a k -sparse band-limited signal from a small collection of noisy time-domain samples. Exploiting a new analytic framework for hierarchical frequency decomposition that performs systematic noise cancellation, we give the first polynomial-time algorithm with a provable (3 + √ 2 + ε ) -approximation guarantee for continuous interpolation. Our method breaks the long-standing C &gt; 100 barrier set by the best previous algorithms, sharply reducing the gap to optimal recovery and establishing a new state of the art for high-accuracy band-limited interpolation. We also give a refined 'shrinking-range' variant that achieves a ( √ 2 + ε + c ) -approximation on any sub-interval (1 -c ) T for some c ∈ (0 , 1) , which gives even higher interpolation accuracy.

## 1 Introduction

The fast Fourier transform (FFT) (Cooley and Tukey, 1965) stands as a cornerstone across engineering, signal processing, mathematics, and theoretical computer science, underpinning both theoretical advances and practical applications. Over time, numerous FFT variants have been proposed, tailored to different signal domains and time-invariance assumptions (Oppenheim et al., 1997; Osgood, 2002; Oppenheim, 2011). In this work we focus on the sparse Fourier transform (SFT), which assumes the signal is band-limited (or Fourier-sparse), i.e., it is observed in the time domain (either discrete or continuous) but is k -sparse in the frequency domain, i.e., its spectrum ̂ x contains only k non-zero components. Our main results concern one-dimensional continuous signals, though we discuss extensions to higher dimensions and the discrete setting. Formally, the band-limited signal is defined as follows.

Definition 1.1 (Band-limited signal) . Let k ∈ Z &gt; 0 . Let δ f i ( f ) denote the Dirac function centered at f i ∈ R . We define the k -sparse band-limited signal x ∗ ( f ) to be as follows:

where v j ∈ C is the coefficient and f j ∈ F is the frequency contained in the frequency range F ⊂ R for each j ∈ [ k ] . We use K to denote the set of f j 's.

<!-- formula-not-decoded -->

Band-limited signals are ubiquitous in practice, underpinning tasks such as image compression and analysis (Watson, 1994), compressed sensing (Donoho, 2006), and (deep) learning with frequencyinvariant kernels (Mei et al., 2021). Fourier methods have also emerged as a powerful tool in machine

learning, inspiring diverse models such as random features for kernel approximation (Rahimi and Recht, 2007), Fourier neural operators for parametric PDEs (Li et al., 2021b), and spectral methods for temporal domain generalization (Yu et al., 2025). Efficient algorithms for computing sparse Fourier transforms are thus of fundamental importance to both signal processing and modern machine learning pipelines. A canonical challenge in this context is band-limited signal interpolation (Chen et al., 2016)-closely related to the 1 Fourier Set-Query problem (Price, 2011)-which seeks to reconstruct (part of) a signal from only a handful of noisy samples (observations x ∗ ( t i ) + g ( t i ) at chosen time points t i ∈ [0 , T ] ), ideally on the order of k , taken from the time domain [0 , T ] d . The band-limit F constrains all frequencies to lie in [ -F, F ] . We give the formal definition as follows.

<!-- formula-not-decoded -->

Definition 1.2 (Band-limited signal interpolation) . Assume that the orignal signal x ∗ ( t ) is k -sparse band-limited. Given the observations of the form x ∗ ( t ) + g ( t ) where g is an arbitrary noisy function, with the signal-to-noise ratio bounded below by a constant (e.g., ∥ x ∗ ∥ T ≳ ∥ g ∥ T , where ∥ x ∥ 2 T := T -d ∫ [0 ,T ] d | x ( t ) | 2 d t denotes the signal's energy ), the goal is to output the reconstructed signal y ( t ) such that where C &gt; 0 is constant and δ ∈ (0 , 1) is an accuracy parameter.

̸

In general, this kind of sparse recovery problems has a long history in signal-processing and computer science (Cooley and Tukey, 1965; Reynolds, 1989; Aibinu et al., 2008; Voelz, 2011; Hassanieh et al., 2012a,b; Ghazi et al., 2013; Indyk and Kapralov, 2014; Indyk et al., 2014; Boashash, 2015; Kapralov, 2016, 2017; Kapralov et al., 2019; Nakos et al., 2019; Jin et al., 2023; Song et al., 2023). A fundamental fact, pointed out in Moitra (2015), is that when the frequency gap is small ( η := min i = j ∈ [ k ] | f i -f j | &lt; 1 /T ), exact recovery of the signal is informational-theoretically impossible. Complementing this negative result, Price and Song (2015) gave a k · polylog( k, F T/δ ) -time δ -error reconstruction algorithm for one-dimensional signals where F is the band-limit, assuming the time domain satisfies T &gt; Ω(log 2 ( k/δ ) /η ) , and that the frequency gap η is known. Chen et al. (2016) strengthened this result by showing that even if the frequency gap is unknown, approximate reconstruction of one-dimensional signals in poly( k, log( FT )) -samples and time is possible, in the sense that the output signal is close to the original signal in the time domain albeit with worse sparsity in the frequency domain 2 . Subsequent works (Chen and Price, 2019b,a; Li et al., 2021a) have improved this result, both in sample-complexity and decoding time. Recently, Li et al. (2021a) improved the sparsity of the output signal from poly( k ) to k · poly log( k ) , settling for a somewhat weaker notion of approximation 3 than that of Chen et al. (2016).

Unfortunately, despite this steady algorithmic progress, the approximation ratio achieved by all prior band-limited interpolation methods has stubbornly remained above a large constant factor (around 100 ). This gap is not merely an artifact of loose analysis: it stems from a fundamental 'triangle-inequality bottleneck'-the three dominant error sources (frequency truncation, polynomial approximation, and linear regression) accumulate additively, and each was previously controlled only up to a constant factor. Closing this gap is crucial both for theory to understand the true limits of sparse recovery under noise and for practice, where large constant blow-ups translate into prohibitively low signal-to-noise requirements.

Our contributions. We break this long-standing barrier and obtain the first high-accuracy algorithm whose approximation factor is strictly below 5 . Concretely, for any k -sparse band-limited signal observed over [0 , T ] and frequency domain F = [ -F, F ] with additive noise g ( t ) , our main result guarantees a reconstructed signal y ( t ) satisfying

<!-- formula-not-decoded -->

for arbitrary accuracy parameters ε, δ ∈ (0 , 1) , using only poly( k, ε -1 , log( FT/δ )) samples and nearly-linear time. Here, the approximation ratio C = 3 + √ 2 + ε quantifies the multiplicative

1 Classical work on band-limited interpolation typically first estimates frequencies and then magnitudes. The latter step can be cast as a Set-Query problem: given a collection of locations, recover the Fourier coefficients ̂ x at those positions. When the frequencies lie on a lattice, the two formulations coincide.

2 More precisely, the error guarantee is ∥ y ( t ) -x ∗ ( t ) ∥ T ≤ O ( ∥ g ( t ) ∥ T + δ ∥ x ∗ ( t ) ∥ T ) , where x ∗ ( t ) is the original signal, y ( t ) is the reconstructed signal, and g ( t ) is the noise distribution.

3 ∥ y ( t ) -x ∗ ( t ) ∥ (1 -c ) T ≤ poly(log( k/cδ )) · ∥ g ( t ) ∥ T + δ ∥ x ∗ ∥ T .

factor by which the reconstruction error ∥ y -x ∗ ∥ T can exceed the unavoidable noise floor ∥ g ∥ T ; it is the standard measure of solution quality in approximation algorithms for signal recovery. This sharply improves the best previous constant C &gt; 100 of Chen et al. (2016), narrows the gap to the information-theoretic optimum of 1 , and resolves an open question posed by Li et al. (2021a). We also give a refined 'shrinking-range' variant that achieves a ( √ 2 + ε + c ) -approximation on any sub-interval (1 -c ) T with the same sample complexity.

Technical novelties. Our improvement hinges on two new analytic ingredients.

1. Ultra-high-sensitivity frequency estimation. We design a filter family that amplifies each true cluster's energy while canceling an equal-scale portion of the adversarial noise. This raises the recoverable energy threshold from Θ(1) to (1 + √ ε ) , eliminating an entire factor of 2 in the first error component.
2. Hierarchical noise-cancellation analysis. We view band-limited interpolation through a unified two-step lens-frequency estimation followed by signal estimation-and track the flow of noise across levels. A refined coupling argument shows that the filtered noise passed to the second step is correlated with the unrecoverable signal energy; bounding them jointly yields the multiplicative factor (3 + √ 2) instead of the additive sum of three constants.

Beyond improving the approximation ratio, our framework is conceptually modular: swapping in any future advances in either sub-routine immediately propagates to a tighter end-to-end guarantee. We believe the tools introduced here-particularly the systematic noise-cancellation bound and the lattice-frequency viewpoint-will be valuable well beyond the specific interpolation task, offering a blueprint for pushing other sparse-recovery algorithms past long-standing constant-factor barriers.

## 1.1 Main results

Recall that all existing algorithms for band-limited interpolation achieve only coarse error bounds of the form

<!-- formula-not-decoded -->

with a constant ≈ 100 in the best published result (Chen et al., 2016); repeated uses of the triangle inequality prevent C from dropping below 3 . We introduce three new ingredientssharper noise control , an ultra-sensitive frequency estimator, and an efficient signal-estimation routine-and combine them in a refined error analysis that collapses these additive losses into a single multiplicative term. This yields the first algorithm with a provable approximation constant strictly below 5 , and it remains near-optimal in both sample complexity and running time.

Our first result uses a sharper error analysis improves C to 3 + √ 2 + ε for any small ε &gt; 0 , which is stated as follows.

Theorem 1.3 (High-accuracy band-limited interpolation, informal version of Theorem H.42) . Let x ∗ ( t ) be a k -sparse band-limited signal with frequencies in [ -F, F ] . Assume the minimum frequency separation is η ≥ Ω(1 /T ) and the signal-to-noise ratio satisfies ∥ x ∗ ∥ T ≳ ∥ g ∥ T . Given observations x ( t ) = x ∗ ( t ) + g ( t ) in time duration [0 , T ] , where g is arbitrary noise. For ε, δ ∈ (0 , 1) , there exists an algorithm that uses poly( k, ε -1 , log(1 /δ )) log( FT ) samples and runtime, and outputs a poly( k, ε -1 , log(1 /δ )) -sparse band-limited signal y ( t ) such that, with high probability,

<!-- formula-not-decoded -->

Our techniques extend to a 'shrinking-range' variant that attains an even tighter constant on any sub-interval (1 -c ) T , which leverages additional 'spatial slack' to lower the approximation ratio to √ 2 + ε + c .

Theorem 1.4 (Ultra-high-accuracy band-limited interpolation with shrinking range, informal version of Theorem I.4) . Let x ∗ ( t ) be a k -sparse band-limited signal with frequencies in [ -F, F ] . Assume the minimum frequency separation is η ≥ Ω(1 /T ) and the signal-to-noise ratio satisfies ∥ x ∗ ∥ T ≳ ∥ g ∥ T . Given observations x ( t ) = x ∗ ( t ) + g ( t ) in time duration [0 , T ] , where g is arbitrary noise. Let T ′ = T (1 -c ) . For ε, δ ∈ (0 , 1) , there exists an algorithm that uses poly( k, ε -1 , log(1 /δ )) log( FT ) samples and poly( k, ε -1 , c -1 , log(1 /δ )) · log 2 ( FT ) runtime, and

outputs a poly( k, ε -1 , c -1 , log(1 /δ )) -sparse band-limited signal y ( t ) such that, with high probability,

<!-- formula-not-decoded -->

We remark that Li et al. (2021a) obtain a related but incomparable guarantee. For any δ &gt; 0 , their algorithm outputs a reconstruction y ( t ) satisfying

∥ y ( t ) -x ∗ ( t ) ∥ (1 -c ) T ≤ α ∥ g ( t ) ∥ T + δ ∥ ̂ x ∗ ( f ) ∥ 1 , where α is the approximation ratio, where c ∈ (0 , 1) is the shrinking parameter, α = poly(log( k/ ( δc ))) , and ̂ x ∗ ( f ) is the Fourier transform of x ∗ ( t ) .. The procedure uses poly( k, log(1 /δ )) log( FT ) samples and poly( k, c -1 , log(1 /δ )) log 2 ( FT ) time, returning a k, poly(1 /c, log( k/δ )) -sparse signal. Hence their result achieves near-optimal sparsity, but the approximation factor grows polylogarithmically with k , 1 /δ , and 1 /c , whereas our algorithm attains a constant ( √ 2 + ε + c ) ratio.

## 1.2 Notations

For any positive integer n , we use [ n ] to denote { 1 , 2 , · · · , n } . We use i to denote √ -1 . For a complex number z ∈ C where z = a + i b and a, b ∈ R . We use z to denote the complex conjugate of z , i.e., z = a -i b . Then it is obvious that | z | 2 = z · z = a 2 + b 2 .

̂ We define our discrete norm as ∥ g ( t ) ∥ 2 S = 1 | S | ∑ t ∈ S | g ( t ) | 2 for function g . We define our weighted discrete norm as ∥ g ( t ) ∥ 2 S,w = ∑ t ∈ S w t | g ( t ) | 2 for function g . We define the continuous T -norm as ∥ g ( t ) ∥ 2 T = 1 T ∫ T 0 | g ( t ) | 2 d t for function g .

We use f ≲ g to denote that there exists a constant C such that f ≤ Cg , and f ≂ g to denote f ≲ g ≲ f . We use ˜ O ( f ) to denote f log O (1) ( f ) . We say x ( t ) is a k -sparse band-limited when x ( t ) = ∑ k j =1 v j exp(2 π i f j t ) . Weuse ̂ x ( f ) to denote the Fourier transform of x ( t ) . More specifically, x ( f ) = ∫ ∞ -∞ x ( t ) exp( -2 π i ft )d t .

In general, we assume x ∗ ( t ) is our ground truth and is a k -sparse band-limited signal. We can observe function x ( t ) = x ∗ ( t ) + g ( t ) for g ( t ) being a noise function. We can observe x ( t ) in duration [0 , T ] . The ground truth x ∗ ( t ) has frequencies in [ -F, F ] .

## 2 Technical Overview

Section 2.1 introduces the framework of discrete Fourier set query. Section 2.2 shows how to apply our discrete Fourier set query estimation algorithm to obtain a high-accuracy band-limited interpolation algorithm as in Theorem 1.3 and Theorem 1.4.

## 2.1 Discrete Fourier Set Query

Many signal processing tasks can be phrased as set query problems in different domains. For instance, the sparse Fourier transform examines only the coefficients at a small set of frequencies-exactly a set-query problem in the frequency domain. Another example is recovering the actual Fourier coefficients when their support (the set of non-zero frequencies) is known.

For concreteness we restrict attention to a one-dimensional discrete signal x ( t ) of length n , written as x ( t ) = ∑ n j =1 ̂ x j e 2 π i jt/n , t ∈ [ n ] . Given a k -subset S ⊂ [ n ] , the set-query task is to recover ̂ x S . Equivalently, define x S ( t ) = ∑ f ∈ S x f e 2 π i ft/n ; this is a k -sparse signal.

̂ Algorithm overview. Our algorithm can be viewed as a three-step pipeline, which collapses into three concrete stages in practice:

1. Uniform sketching. Any discrete k -sparse signal has energy bound R = k , meaning sup t | x ( t ) | 2 ≤ R ∥ x ∥ 2 T . Consequently, a uniform sample S 0 ⊂ [ n ] of size O ( k log k ) already preserves the signal's energy up to a constant factor and forms a faithful oblivious sketch.

2. Sketch distillation. We refine S 0 via the Sketch Distillation procedure: a well-balanced sampler chooses a linear-size subset S 1 ⊂ S 0 and weights w such that ∥ x S ∥ S 1 ,w ≈ ∥ x S ∥ 2 /n for every signal supported on S , while simultaneously ensuring the weighted energy of the orthogonal component x S + g is not amplified, i.e. ∥ x S + g ∥ S 1 ,w = O ( ∥ x S + g ∥ T ) .

̂ ̂ Together, these steps give a linear-sample, high-accuracy solution to the discrete Fourier set-query problem, achieving (1 + ε ) approximation with high probability. A routine analysis gives an O (1) approximation; below we refine it to (1 + ε ) .

3. Weighted regression. With samples { x ( t ) } t ∈ S 1 and weights w , we solve the weighted leastsquares problem min v ′ ∈ C k ∥ ∥ √ w ◦ ( Av ′ -b ) ∥ ∥ 2 , where A i,j = e 2 π i f j t i /n and b i = x ( t i ) . The solution x S ( f j ) = v ′ j yields a reconstruction whose error obeys ∥ x S -x S ∥ T ≤ ε ∥ x S + g ∥ T .

## Algorithm 1 Discrete 1-D Fourier Set-Query

```
1: procedure SETQUERY( x, n, k, S, ε ) 2: { f 1 , . . . , f k } ← S /* Step 1: Uniform sketching */ 3: S 0 ← Sample O ( ε -2 k log k ) points i.i.d. from Uniform([ n ]) /* Step 2: Sketch distillation */ 4: F ← {∑ k j =1 v j e 2 π i f j t/n ∣ ∣ v j ∈ C } 5: ( { t 1 , . . . , t s } , w ) ← RANDBSS+ ( k, F , Uniform( S 0 ) , ( ε/ 4) 2 ) ▷ Algorithm 3 /* Step 3: Weighted regression */ 6: for ( i, j ) ∈ [ s ] × [ k ] do 7: A i,j ← e 2 π i f j t i /n 8: end for 9: for i ∈ [ s ] do 10: b i ← x ( t i ) ▷ observe x at t i 11: end for 12: v ′ ← arg min v ′ ∈ C k ∥ ∥ √ w ◦ ( Av ′ -b ) ∥ ∥ 2 13: return v ′ 14: end procedure
```

Composition of well-balanced samplers. To obtain a (1 + ε ) guarantee we must show that the final sketch ( S 1 , w ) arises from a single well-balanced sampling procedure (WBSP). In general, composing two WBSPs may violate well-balancedness: while the first property (accurate energy estimation for every f ∈ F ) is preserved, the second property concerning weight sum and condition number can fail.

Our setting is special: the first sampler draws each point uniformly from [ n ] . We prove that, under this choice and with the tight energy bound R = k for band-limited signals, each sample produced by the two-stage composition is distribution-equivalent to a fresh uniform draw. Hence the composite sampler is itself well-balanced, allowing us to invoke the sharper error analysis and conclude that

∥ ̂ y S -̂ x S ∥ 2 ≤ ε ∥ ̂ x S ∥ 2 2 with high probability. Thus we obtain a linear-sample, high-accuracy algorithm for the discrete Fourier set-query problem.

## 2.2 High-accuracy band-limited interpolation

In this section, we introduce how to obtain a high-accuracy one-dimensional band-limited interpolation algorithm (Theorem 1.3), which improves the constant-accuracy algorithm by Chen et al. (2016).

Let us briefly summarize the previous algorithm in Chen et al. (2016). The high-level idea is to first find some small intervals in the frequency domain such that each contains some significant frequencies of the signal x ∗ . (These intervals are called 'heavy-clusters' in their paper.) Then, they use some filter techniques (also used in Price and Song (2015)) to reduce the problem of reconstructing x ∗ , a signal with multiple heavy-clusters to several single heavy-cluster signals. Then, for each single

heavy-cluster signal, since the band-limit is small, they can efficiently estimate its frequencies. Finally, they reconstruct a poly( k ) -sparse signal that is close to x ∗ via a robust polynomial learning algorithm. More specifically, their algorithm consists of the following steps:

1. They show that the ground-truth signal x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t can be approximated by x S ( t ) = ∑ j ∈ S v j e 2 π i f j t , where S := { j ∈ [ k ] : f j ∈ some heavy-cluster C i } is the set of frequencies in the heavy-clusters. This step will cause an approximation error E 1 := ∥ x ∗ -x S ∥ T ≤ 1 . 2 N 4 , where N 2 := ∥ g ∥ 2 T + δ ∥ x ∗ ∥ 2 T appears in the approximation error of the band-limited interpolation problem.
2. They solve a Frequency Estimation problem for x S using the filter techniques and multipleto-one heavy-cluster reduction, and get a list L of O ( k ) candidate frequencies so that for each j ∈ S , f j is close to some f p j ∈ L .
4. It remains to reconstruct x S, poly ( t ) , which is a variant of Signal Estimation problem. They use a sampling-and-regression approach to obtain a poly( k ) -sparse signal y ( t ) with an approximation error E 3 := ∥ y -x S, poly ∥ T ≤ 2200 N .
4. ˜ 3. The signal x S ( t ) can be decomposed into ∑ | L | i =1 e 2 π i ˜ f i t · x ∗ i ( t ) , where x ∗ i ( t ) := ∑ j : p j = i e 2 π i ( f j -˜ f i ) is a one-cluster signal with small band-limit. For each x ∗ i ( t ) , they prove that there exists a low-degree polynomial P i ( t ) that can approximate it. Let's denote the polynomial-approximated signal ∑ | L | i =1 e 2 π ˜ f i t · P i ( t ) by x S, poly ( t ) , which has an approximation error E 2 := ∥ x S -x S, poly ∥ T ≤ δ ∥ x S ∥ T . 5

By triangle inequality, the total approximation error is ∥ y -x ∗ ∥ T ≤ E 1 + E 2 + E 3 ≤ C · N , where C &gt; 1000 is an absolute constant.

3-approximation barrier To achieve high-accuracy band-limited interpolation, we need to control the errors E 1 , E 2 , and E 3 .

- For E 1 , it is coupled with the second step, since the approximation error of x S is connected to the significance of each heavy-cluster. If we choose a too-small E 1 , x S will contain some not-so-significant frequencies, and the Frequency Estimation algorithm may not be able to find them. Thus, with the techniques in Chen et al. (2016), we cannot make E 1 to be less than N . Even worse, due to the noise g in the observed signal, the error will be at least 2 N .
- For E 2 , it only depends on the error parameter δ . Since the sample and time complexities of the algorithm only depend logarithmically on 1 /δ , it allows us to re-scale δ and make E 2 very small.
- For E 3 , where we lose a big constant, we need a high-accuracy Signal Estimation algorithm to recover the polynomial-approximated signal x S, poly ( t ) . However, as we discussed in the previous sections, the error of the Signal Estimation will be at least N .

Therefore, there is a 3-approximation barrier in Chen et al. (2016)'s approach due to the triangle inequality.

In the remainder of this section, we first introduce our techniques to achieve a (7 + ε ) -approximation error. Then, we show how to overcome the barrier and achieve a (3 + √ 2 + ε ) -approximation error.

## 2.2.1 (7 + ε ) -approximation algorithm

High sensitivity frequency estimation We first improve E 1 from 1 . 2 N to (1 + ε ) N by proposing a high sensitivity frequency estimation method. More specifically, to identify the heavy-clusters of

4 Due to the noisy observations, not every frequency in the heavy-clusters is recoverable. This gap causes an extra implicit error term in Chen et al. (2016), which is about 12 N . Section 2.2.1 has a more detailed discussion.

5 We remark that even if the ground-truth signal x ∗ ( t ) can be well-approximated by a mixed Fourierpolynomial signal ˜ x ( t ) , we are unable to recover every basis of ˜ x ( t ) due to the limitation of the frequency estimation procedure. Thus, directly applying linear regression to partially reconstruct ˜ x ( t ) will not guarantee to be a (1 + ε ) -approximation of x ∗ ( t ) .

the signal x ∗ , Chen et al. (2016) designed a filter function H such that H · x ∗ has high energy in each heavy cluster C i ; that is,

Moreover, H 's frequencies are contained in a small interval of length ∆ . These two properties imply that for any true frequency f i ∈ C i , the signal H · x ∗ with frequency domain restricted to [ f i -∆ , f i +∆] is a one heavy-cluster signal with small band-limit, which allows us to use Price and Song (2015)'s approach to estimate f i . The filter function H in Chen et al. (2016) is only O (1) -sensitive, which means it can concentrate a constant fraction of the signal's energy. And for those less important frequencies, they cannot be clustered by H and will be lost in the frequency estimation procedure.

<!-- formula-not-decoded -->

We manage to modify their filter construction and obtain a (1 -ε ) -sensitive filter function H such that the signal x S ∗ consisted of the frequencies in the new heavy-clusters has about (1 -ε ) -fraction of energy of x ∗ . More specifically, we have

<!-- formula-not-decoded -->

To prove that we can actually estimate the frequencies in the new heavy-clusters, we observe a subtle point: the energy condition of heavy-cluster and the energy condition of frequency estimation are inconsistent due to the noise in observations. To be able to estimate the one heavy-cluster signal's frequency, it is required that which is different from Eq. (1). In other words, not all frequencies in S ∗ are recoverable, but only most of them. Since Chen et al. (2016) only wants a constant approximation error, they may simply ignore this difference by losing a constant factor in accuracy. For us, however, we need to make it precise. We define S to be a subset of S ∗ containing the frequencies in the heavy-clusters satisfying Eq. (2). We analyze the effect of H · g and show that by strengthening the RHS of heavy-cluster's energy condition (Eq. (1)) to 4 T k N 2 , we can bound the unrecoverable part's energy by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the recoverable part x S , we can just follow Chen et al. (2016)'s approach to estimating the frequencies in each heavy-cluster.

Generalized high-accuracy signal estimation We apply our three-step Fourier set-query framework to solve the signal estimation problem in the forth step of Chen et al. (2016)'s algorithm and improve E 3 from 2200 N to (4 + ε ) N . We first define the problem more formally. By frequency estimation, we obtain a list of candidate frequencies of x S and in the third step, we know that it can be approximated by x S, poly ( t ) := ∑ | L | i =1 e 2 π i ˜ f i t · P i ( t ) with very tiny error E 2 , where P i ( t ) are some degreed polynomials. We can rewrite x S, poly in the Fourier-monomial mixed basis :

where v i,j ∈ C and ˜ f i ∈ L are known. That is, we need to learn { v i,j } given noisy observations x S, poly ( t ) + g ′ ( t ) , which is a Signal Estimation problem for the following family of signals:

<!-- formula-not-decoded -->

We apply our three-step framework as follows.

<!-- formula-not-decoded -->

- In Step 1, we first need an energy bound F mix . Chen et al. (2016) showed that

Then we can get that uniformly sample ˜ O ( | L | 4 d 4 ε -1 ) points in [0 , T ] gives an oblivious sketching for x S, poly . Furthermore, we can show that this sampler is ε -well-balanced.

<!-- formula-not-decoded -->

- In Step 2, since we aim at achieving high-accuracy, we do not distill the sketch but directly apply the sharper error analysis to control the energy of the orthogonal part of noise.
- In Step 3, we solve a weighted linear regression to estimate the coefficients and obtain a signal y ′ ( t ) ∈ F mix such that

<!-- formula-not-decoded -->

Then, we can transform y back to a poly( k ) -sparse signal with error ∥ y -y ′ ∥ T ≤ E 2 .

Combining them together and re-scaling ε and δ , we get that

<!-- formula-not-decoded -->

Therefore, we obtain a band-limited interpolation algorithm with (7 + ε ) -approximation error.

<!-- image -->

1

Figure 1: An illustration of the signal-noise cancellation effect (Eq. (3)). In (a), the blue region corresponds to the first term of Eq. (3), which roughly equals the energy of x -x S ∗ 1 . In (b), the red and blue regions correspond to the second term, where most of the energy is canceled. Thus, their total energy is very close to ∥ x -x S ∗ 1 ∥ T .

## 2.2.2 (3 + √ 2 + ε ) -approximation algorithm

How can we further improve this algorithm? We observe that E new 3 can be written more precisely as (1 + ε ) ∥ g ∥ T +(3 + ε ) N . On the other hand, E new 1 and E 1 . 5 only depend on N . If we can take a smaller value for N 2 , i.e., ε ( ∥ g ∥ 2 T + δ ∥ x ∥ 2 T ) , then we will improve approximation error. We show that it is possible via an ultra-high sensitivity frequency estimation method .

Ultra-high sensitivity frequency estimation To improve the sensitivity of the frequency estimation method, let N 2 1 := ε N 2 and consider the heavy-clusters with parameter N 1 . Let S ∗ 1 denote the set of frequencies of x ∗ ( t ) in the N 1 -heavy-clusters. By the same analysis as in our previous frequency estimation approach, we have E new + 1 := ∥ x ∗ -x S ∗ 1 ∥ T ≤ (1 + ε ) N 1 .

However, due to the inconsistent energy conditions, only those frequencies in the heavy-clusters satisfying Eq. (2) are recoverable. Let S 1 denote the set of such frequencies, and we need to upper bound ∥ x S ∗ 1 -x S 1 ∥ T . Previously, we strengthen the heavy-cluster's condition (Eq. (1)) and get a E 1 . 5 ≤ (2 + ε ) ∥ g ∥ T bound. Here, instead, we relax the RHS of Eq. (2) to ε · T k N 2 1 . Intuitively, more frequencies will satisfy the new frequency estimation condition; and if there is a unrecoverable frequency f ∗ ∈ S ∗ 1 \ S 1 , it indicates that its contribution in filtered signal H · x ∗ is cancelled out by the filtered noise H · g . Using this signal-noise cancellation effect , we prove that:

<!-- formula-not-decoded -->

which saves a factor of 2 from E 1 . 5 by introducing an extra term ∥ H ( x -x S 1 ) ∥ T . Recall ∥ x -x S 1 ∥ T is related to E new 3 , the error of the signal estimation procedure. We can decompose it into the 'passing energy' ∥ H ( x -x S 1 ) ∥ T and 'filtered energy' ∥ (Id -H )( x -x S 1 ) ∥ T and bound them by:

<!-- formula-not-decoded -->

Thus, Eq. (3) can be considered as upper-bounding E 1 . 5 and E new 3 simultaneously. Combining them together, we get the following error guarantee for the frequency recoverable signal x S 1 :

<!-- formula-not-decoded -->

Then, by a more careful analysis of the HASHTOBINS approach used by Chen et al. (2016) for Frequency Estimation, we show that x S 1 's frequencies can be efficiently approximated, which gives an ultra-high sensitivity frequency estimation method.

The remaining part of the algorithm is almost identical to the previous one. We run the high-accuracy signal estimation algorithm to reconstruct x S 1 . Let y ( t ) denote the output band-limited signal. By Eq. (4) and re-scaling ε and δ , we have

<!-- formula-not-decoded -->

Therefore, we achieve a (3 + √ 2 + ε ) -approximation error for the band-limited interpolation.

## 2.2.3 ( √ 2 + ε + c ) -approximation algorithm with shrinking range

When we only care about the signal on the interior window [0 , (1 -c ) T ] , the two 'edge strips' of length cT/ 2 at the beginning and end become disposable budget. We exploit this slack with a shrinking-range filter H that (i) leaves the interior almost unchanged and (ii) suppresses the contribution of every frequency band whose energy is already comparable to the noise. This ensures that, after filtering, all irrecoverable energy cancels with the adversarial noise, so the residual we must approximate is strictly smaller than in the full-range setting.

Using the same high-sensitivity filter construction, but tuned to the relaxed threshold ε 1 ∥ g ∥ 2 T , we identify a set S of 'truly heavy' frequencies inside each cluster. We show that (see Lemma I.1)

<!-- formula-not-decoded -->

Then we can recovers, for every f j ∈ S , an approximation ˜ f j with resolution O (∆ 0 √ ∆ 0 T ) using only poly( k, ε -1 , c -1 ) log( FT ) samples. See Corollary I.2 for more details.

On T ′ = (1 -c ) T we run the high-accuracy signal-estimation routine (Section G.2), but now with the uniform sampler restricted to T ′ . Because the filter already tames boundary energy, a linear-sized well-balanced sample suffices to learn the coefficients with relative error 1 + ε .

Then we can replace each narrow band around ˜ f j by a low-degree polynomial P j ( t ) (degree d = O ( T ∆ 3 / 2 0 + k 3 log k ) ) whose Fourier support stays inside the band and whose time-domain error is O ( δ ) ∥ x S ∥ T .

Putting it together. Summing the four error sources-cluster trimming, frequency rounding, polynomial patching, and weighted regression-and rescaling ε, δ yields Theorem 1.4: for any c ∈ (0 , 1) , with sample complexity and runtime poly ( k, ε -1 , c -1 , log(1 /δ ) ) log O (1) ( FT ) , and output sparsity poly ( k, ε -1 , c -1 , log(1 /δ ) ) . Thus, shrinking the range lets us push the approximation constant all the way down to √ 2 + ε + c .

## 3 Conclusion

In this work, we break the long-standing constant-factor barrier for noisy band-limited interpolation. Our primary algorithm delivers a (3 + √ 2 + ε ) -approximation with poly( k, ε -1 , log(1 /δ )) sample and time complexity. A refined 'shrinking-range' variant pushes the constant down to ( √ 2 + ε + c ) on any interior window (1 -c ) T , demonstrating that additional spatial slack can be traded directly for reconstruction accuracy. Two technical ingredients drive these improvements: We introduce a new filter family simultaneously magnifies cluster energy and cancels adversarial noise, allowing

<!-- formula-not-decoded -->

reliable recovery at an energy threshold arbitrarily close to the information-theoretic optimum. We also proved a unified view of frequency and signal estimation tracks how residual noise propagates through each stage, replacing three additive error terms with a single multiplicative bound and collapsing the approximation constant. These techniques are modular and extend naturally to broader sparse-recovery settings, opening avenues for even tighter guarantees in higher dimensions, alternate transform domains, and streaming environments.

Our guarantees rely on several idealised assumptions: the signal must be exactly k -sparse in the frequency domain (or perfectly lattice-aligned), the signal-to-noise ratio must exceed a constant threshold, and the analysis is fully worked out only in one dimension; relaxing any of these conditions can inflate the hidden polynomial factors in our sample and runtime bounds or even invalidate recovery. Moreover, the ultra-sensitive filters we employ require high numerical precision-round-off error or model mismatch may erode the stated constants in practical deployments. It remains the open whether these assumptions can be relaxed and further improve the approximation ratio.

## Acknowledgment

We thank anonymous NeurIPS reviewers for their constructive comments.

## References

- Aibinu, A. M., Salami, M.-J. E., Shafie, A. A., and Najeeb, A. R. (2008). Mri reconstruction using discrete fourier transform: a tutorial.
- Alman, J. and Williams, V. V. (2021). A refined laser method and faster matrix multiplication. In Proceedings of the 2021 ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 522-539. SIAM.
- Batson, J., Spielman, D. A., and Srivastava, N. (2012). Twice-ramanujan sparsifiers. SIAM Journal on Computing , 41(6):1704-1721.
- Boashash, B. (2015). Time-frequency signal analysis and processing: a comprehensive reference . Academic press.
- Borwein, P. and Erdélyi, T. (2006). Nikolskii-type inequalities for shift invariant function spaces. Proceedings of the American Mathematical Society , 134(11):3243-3246.
- Chen, X., Kane, D. M., Price, E., and Song, Z. (2016). Fourier-sparse interpolation without a frequency gap. In 2016 IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS) , pages 741-750. IEEE.
- Chen, X. and Price, E. (2019a). Active regression via linear-sample sparsification. In Conference on Learning Theory (COLT) , pages 663-695. PMLR.
- Chen, X. and Price, E. (2019b). Estimating the frequency of a clustered signal. In ICALP .
- Chernoff, H. (1952). A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations. The Annals of Mathematical Statistics , pages 493-507.
- Cooley, J. W. and Tukey, J. W. (1965). An algorithm for the machine calculation of complex fourier series. Mathematics of computation , 19(90):297-301.
- Donoho, D. (2006). Compressed sensing. IEEE Transactions on Information Theory , 52(4):12891306.
- Ghazi, B., Hassanieh, H., Indyk, P., Katabi, D., Price, E., and Shi, L. (2013). Sample-optimal average-case sparse fourier transform in two dimensions. In 2013 51st Annual Allerton Conference on Communication, Control, and Computing (Allerton) , pages 1258-1265. IEEE.
- Hassanieh, H., Indyk, P., Katabi, D., and Price, E. (2012a). Nearly optimal sparse fourier transform. In Proceedings of the forty-fourth annual ACM symposium on Theory of computing , pages 563-578.

- Hassanieh, H., Indyk, P., Katabi, D., and Price, E. (2012b). Nearly optimal sparse fourier transform. In Proceedings of the forty-fourth annual ACM symposium on Theory of computing , pages 563-578.
- Indyk, P. and Kapralov, M. (2014). Sample-optimal Fourier sampling in any constant dimension. In IEEE 55th Annual Symposium onFoundations of Computer Science (FOCS) , pages 514-523. IEEE.
- Indyk, P., Kapralov, M., and Price, E. (2014). (nearly) sample-optimal sparse fourier transform. In Proceedings of the twenty-fifth annual ACM-SIAM symposium on Discrete algorithms , pages 480-499. SIAM.
- Jin, Y., Liu, D., and Song, Z. (2023). A robust multi-dimensional sparse fourier transform in the continuous setting. In SODA .
- Kapralov, M. (2016). Sparse fourier transform in any constant dimension with nearly-optimal sample complexity in sublinear time. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing , pages 264-277.
- Kapralov, M. (2017). Sample efficient estimation and recovery in sparse FFT via isolation on average. In Umans, C., editor, 58th IEEE Annual Symposium on Foundations of Computer Science, FOCS 2017, Berkeley, CA, USA, October 15-17, 2017 , pages 651-662. IEEE Computer Society.
- Kapralov, M., Velingker, A., and Zandieh, A. (2019). Dimension-independent sparse Fourier transform. In Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 2709-2728. SIAM, https://arxiv.org/pdf/1902.10633.pdf .
- Kós, G. (2008). Two turán type inequalities. Acta Mathematica Hungarica , 119(3):219-226.
- Le Gall, F. (2014). Powers of tensors and fast matrix multiplication. In Proceedings of the 39th international symposium on symbolic and algebraic computation (ISSAC) , pages 296-303. ACM.
- Lee, Y. T. and Sun, H. (2015). Constructing linear-sized spectral sparsification in almost-linear time. In Guruswami, V., editor, IEEE 56th Annual Symposium on Foundations of Computer Science, FOCS 2015, Berkeley, CA, USA, 17-20 October, 2015 , pages 250-269. IEEE Computer Society.
- Li, J., Liu, A., and Moitra, A. (2021a). Sparsification for sums of exponentials and its algorithmic applications. arXiv preprint arXiv:2106.02774 .
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. (2021b). Fourier neural operator for parametric partial differential equations. In ICLR .
- Mei, S., Misiakiewicz, T., and Montanari, A. (2021). Learning with invariances in random features and kernel models. In Belkin, M. and Kpotufe, S., editors, Conference on Learning Theory, COLT 2021, 15-19 August 2021, Boulder, Colorado, USA , volume 134 of Proceedings of Machine Learning Research , pages 3351-3418. PMLR.
- Moitra, A. (2015). The threshold for super-resolution via extremal functions. In STOC . arXiv preprint arXiv:1408.1681.
- Nakos, V., Song, Z., and Wang, Z. (2019). (nearly) sample-optimal sparse fourier transform in any dimension; ripless and filterless. In 2019 IEEE 60th Annual Symposium on Foundations of Computer Science (FOCS) , pages 1568-1577. IEEE.
- Oppenheim, A. V. (2011). Lecture notes: Fourier transform properties. https://ocw.aprende.org/resources/res-6-007-signals-and-systems-spring-2011/lecturenotes/MITRES\_6\_007S11\_lec09.pdf.
- Oppenheim, A. V., Willsky, A. S., Nawab, S. H., Hernández, G. M., et al. (1997). Signals &amp; systems . Pearson Educación.
- Osgood, B. (2002). Lecture notes for ee 261 the fourier transform and its applications.
- Price, E. (2011). Efficient sketches for the set query problem. In Randall, D., editor, Proceedings of the Twenty-Second Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2011, San Francisco, California, USA, January 23-25, 2011 , pages 41-56. SIAM.

- Price, E. and Song, Z. (2015). A robust sparse Fourier transform in the continuous setting. In 2015 IEEE 56th Annual Symposium on Foundations of Computer Science , pages 583-600. IEEE.
- Rahimi, A. and Recht, B. (2007). Random features for large-scale kernel machines. In NeurIPS , volume 20.
- Reynolds, G. O. (1989). The New Physical Optics Notebook: Tutorials in Fourier Optics. ERIC.
- Song, Z., Sun, B., Weinstein, O., and Zhang, R. (2023). Quartic samples suffice for fourier interpolation. In FOCS . arXiv preprint arXiv:2210.12495.
- Voelz, D. G. (2011). Computational fourier optics: a MATLAB tutorial . SPIE press Bellingham, Washington.
- Watson, A. B. (1994). Image compression using the discrete cosine transform. Mathematica Journal , 4:81-88.
- Williams, V. V. (2012). Multiplying matrices faster than coppersmith-winograd. In Karloff, H. J. and Pitassi, T., editors, Proceedings of the 44th Symposium on Theory of Computing Conference, STOC 2012, New York, NY, USA, May 19 - 22, 2012 , pages 887-898. ACM.
- Yu, E., Lu, J., Yang, X., Zhang, G., and Fang, Z. (2025). Learning robust spectral dynamics for temporal domain generalization. arXiv preprint arXiv:2505.12585 .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the claims made, including the contributions made in the paper and important assumptions and limitations.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We include the limitation discussion in Section 3.

## Guidelines:

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

Justification: All assumptions of this work are made within the statement of theorems or lemmas. For each theoretical result:

- The formal version of Theorem 1.3 is Theorem H.20, where the proof is in Section H.
- The formal version of Theorem 1.4 is Theorem I.4, where the proof is in Section I.

## Guidelines:

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

Justification: The paper does not include experiments.

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

Justification: All authors have reviewed and confirmed that the research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We include the broader impacts discussion in Section J.

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

Justification: The paper does not include experiments and poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

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

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Preliminaries

This section is organized as follows. In Section A.1, we provide some technical tools in probability theory and linear algebra. In Section A.2, we review the Fourier transformation for different types of signals.

## A.1 Tools and inequalities

Lemma A.1 (Chernoff Bound Chernoff (1952)) . Let X 1 , X 2 , · · · , X n be independent random variables. Assume that 0 ≤ X i ≤ 1 always, for each i ∈ [ n ] . Let X = X 1 + X 2 + · · · + X n and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Definition A.2 ( ε -net) . Let T be a metric space with distance measure d . Consider a subset K ⊂ T and let ε &gt; 0 . A subset N ⊆ K is called an ε -net of K if every point in K is within distance ε of some point of N , i.e.

Fact A.3 (Fast matrix multiplication) . We use T mat ( a, b, c ) to denote the time of multiplying an a × b matrix with another b × c matrix.

<!-- formula-not-decoded -->

We use ω to denote the exponent of matrix multiplication, i.e., T mat ( n, n, n ) = n ω . Currently ω ≈ 2 . 373 Williams (2012); Le Gall (2014); Alman and Williams (2021).

<!-- formula-not-decoded -->

Fact A.4 (Weighted linear regression) . Given a matrix A ∈ C n × d , a vector b ∈ C n and a weight vector w ∈ R n &gt; 0 , it takes O ( nd ω -1 ) time to output an x ′ such that where √ W := diag( √ w 1 , . . . , √ w n ) ∈ R n × n , and ω ≈ 2 . 373 is the exponent of matrix multiplication Williams (2012); Le Gall (2014); Alman and Williams (2021).

Fact A.5. For any x ∈ (0 , 1) , we have cos( x ) ≤ exp( -x 2 / 2) .

## A.2 Basics of Fourier transformation

The definition of high dimensional Fourier transform is as follows:

and the definition of high dimensional inverse Fourier transform is as follows:

<!-- formula-not-decoded -->

Note that when we replace d = 1 in the definition of high dimensional Fourier transform and inverse Fourier transform above, we get the definition of one-dimensional Fourier transform and inverse Fourier transform.

<!-- formula-not-decoded -->

The definition of discrete Fourier transform is as follows:

and the definition of discrete inverse Fourier transform is as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Appendix

A continuous k -Fourier sparse signal x ( t ) : R d → C can be represented as follows:

<!-- formula-not-decoded -->

Thus, ̂ x ( f ) is:

A discrete k -Fourier sparse signal x ∈ C n can be represented as follows:

So, ̂ x f is:

## B Definitions of Semi-Continuous Fourier Set Query and Interpolation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In this section, we give the formal definitions of the problems studied in this paper. In Section B.1, we define the Fourier set query for discrete and continuous signals. In Section B.2, we define the band-limited interpolation problem and its two sub-problems: frequency estimation and signal estimation. And in Section B.3, we discuss the importance sampling method.

## B.1 Formal definitions of Fourier set query

The discrete Fourier set query problem is defined as follows:

Definition B.1 (Discrete Fourier set query problem) . Let x ∈ C n and ̂ x be its discrete Fourier transformation. Let ε &gt; 0 . Given a set S ⊆ [ n ] and query access to x , the goal is to use a few queries to compute a vector x ′ with support supp( x ′ ) ⊆ S such that

We also define the continuous Fourier set query problem as follows:

Definition B.2 (Continuous Fourier set query problem) . For d ≥ 1 , let x ∗ ( t ) be a signal in time duration [0 , T ] d . Let ̂ x ∗ ( f ) denote the continuous Fourier transformation of x ∗ ( t ) . Let ε &gt; 0 . Given a set S ⊆ R d of frequencies such that supp( ̂ x ∗ ) ⊆ S , and observations of the form x ( t ) = x ∗ ( t )+ g ( t ) , where g ( t ) denotes the noise. The goal is to output a Fourier-sparse signal x ′ ( t ) with support supp( ̂ x ′ ) ⊆ S such that ∥ x ′ -x ∗ ∥ 2 T ≤ (1 + ε ) · ∥ g ∥ 2 T .

<!-- formula-not-decoded -->

## B.2 Formal definitions of semi-continuous band-limited interpolation

In this section, we provide the following formal definition of the semi-continuous band-limited interpolation problem, where we assume that the frequencies of the signal are contained in a lattice. Problem B.3 (Semi-continuous band-limited interpolation problem) . Given a basis B of m known vectors b 1 , b 2 , · · · b m ∈ R d , let Λ( B ) ⊂ R d denote the lattice

̸

Suppose that f 1 , f 2 , · · · , f k ∈ Λ( B ) , ∀ i ∈ [ k ] , | f i | ≤ F . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i ⟨ f j ,t ⟩ , and let g ( t ) denote the noise. Given observations of the form x ( t ) = x ∗ ( t ) + g ( t ) , t ∈ [0 , T ] d . Let η = min i = j ∥ f j -f i ∥ ∞ . There are three goals:

<!-- formula-not-decoded -->

1. The first goal is to design an algorithm that output f 1 , f 2 , · · · , f k exactly given query access to the signal x ( t ) for t ∈ [0 , T ] d .
2. The second goal is to design an algorithm that output a set L of frequencies such that, for each f i , there is f ′ i ∈ L , ∥ f i -f ′ i ∥ 2 ≤ D/T .

Then, we extract two sub-problems from Problem B.3: Frequency Estimation and Signal Estimation. We give their definitions below.

3. The third goal is to design an algorithm that output y ( t ) = ∑ ˜ k j =1 v ′ j · e 2 π i f ′ j t such that ∫ [0 ,T ] d | y ( t ) -x ( t ) | 2 d t ≲ ∫ [0 ,T ] d | g ( t ) | 2 d t .

We first define the d -dimensional frequency estimation under the semi-continuous as follows. In this problem, we want to recover each frequencies in a small range.

Problem B.4 (Frequency estimation) . Given a basis B of m known vectors b 1 , b 2 , · · · b m ∈ R d , let Λ( B ) ⊂ R d denote the lattice

̸

The goal is to design an algorithm that output a set L of frequencies such that, for each f i , there is f ′ i ∈ L , ∥ f i -f ′ i ∥ 2 ≤ D/T .

Suppose that f 1 , f 2 , · · · , f k ∈ Λ( B ) . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i ⟨ f j ,t ⟩ , and let g ( t ) denote the noise. Given observations of the form x ( t ) = x ∗ ( t ) + g ( t ) , t ∈ [0 , T ] d . Let η = min i = j ∥ f j -f i ∥ ∞ .

We remark that the recovered frequencies in L are not necessary to be in Λ( B ) , and D is a parameter that can depend on k .

Next, we define the d -dimensional Signal Estimation under the semi-continuous setting as follows. In this problem, we want to recover a signal that can approximate the ground-truth signal in the time domain.

Problem B.5 (Signal Estimation problem) . Given a basis B of m known vectors b 1 , b 2 , · · · b m ∈ R d , let Λ( B ) ⊂ R d denote the lattice

̸

Suppose that f 1 , f 2 , · · · , f k ∈ Λ( B ) . Let x ∗ ( t ) := ∑ k j =1 v j e 2 π i ⟨ f j ,t ⟩ , and let g ( t ) denote the noise. Given observations of the form x ( t ) = x ∗ ( t ) + g ( t ) , t ∈ [0 , T ] d . Let η = min i = j ∥ f j -f i ∥ ∞ .

<!-- formula-not-decoded -->

The goal is to design an algorithm that outputs y ( t ) = ∑ ˜ k j =1 v ′ j · e 2 π i f ′ j t such that

Note that outputting y ( t ) = ∑ ˜ k j =1 v ′ j · e 2 π i f ′ j t means outputting { v ′ j , f ′ j } j ∈ [ ˜ k ] .

<!-- formula-not-decoded -->

Remark B.6. We note that given the solution of Frequency Estimation (Problem B.4), Signal Estimation (Problem B.5) can be formulated as a Fourier set query problem (Problem B.2). More specifically, by Frequency Estimation, we will find a set that contains all frequencies of the ground truth signal x ∗ ( t ) . Then, we only need to recover the coefficients with frequencies in this set, which is equivalent to a set query problem.

## B.3 Facts about importance sampling

Important sampling try to estimate a statistic value in one distribution by taking samples in another distribution. In particular, Chen and Price (2019a) considered the importance sampling for estimating the norm of functions in a linear family F .

In this followings, we first provide some basic definitions about linear function family.

<!-- formula-not-decoded -->

Definition B.7 (Condition number of sampling distribution) . Let G be any domain and F is a linear function family from G to C . Let D be an arbitrary distribution over G . Then the condition number of D with respect to F is defined as follows:

where

<!-- formula-not-decoded -->

Definition B.8 (Orthonormal basis for linear function family) . Let G be any domain. Given a linear function family F from G to C , and a probability distribution D over G . We say { v 1 , . . . , v d } form an orthonormal basis of F with respect to D , if they satisfy the following properties:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- for any f ∈ F , f ∈ span { v 1 , . . . , v d } .

For an unknown function f ∈ F , the goal of importance sampling is to estimate ∥ f ∥ D , given samples from another distribution D ′ . The following definition introduces the importance sampling procedure and condition number of the importance sampling distribution.

Definition B.10 (Definition 3.1 of Chen and Price (2019a)) . For any unknown distribution D ′ over the domain G and any function f ∈ F , let f ( D ′ ) ( t ) := √ D ( t ) D ′ ( t ) · f ( t ) be the importance sampling function for some known distribution D such that

Then, we can use samples from D ′ to estimate ∥ f ( D ′ ) ∥ D ′ , which gives an estimate of ∥ f ∥ D .

<!-- formula-not-decoded -->

When the family F and D is clear, we use K IS ,D ′ to denote the condition number of importance sampling from D ′ :

From Definition B.10, we know that the efficiency of importance sampling depends on how many samples we need to estimate ∥ f D ′ ∥ D ′ . The following lemma provide a criteria for judging whether a set of samples gives a good estimation for the norm of function.

<!-- formula-not-decoded -->

Lemma B.11 (Lemma 4.2 in Chen and Price (2019a)) . For any ε ∈ (0 , 1) , let S = { t 1 , . . . , t s } and the weight vector w ∈ R s &gt; 0 . Define a matrix A ∈ R s × d be the s × d matrix defined as A i,j = √ w i · v j ( t i ) , where { v 1 , . . . , v d } is an orthonormal basis for F . Then if and only if the eigenvalues of A ∗ A are in [1 -ε, 1 + ε ] .

<!-- formula-not-decoded -->

The following lemma shows that the sample complexity depends on the condition number K IS ,D ′ : Lemma B.12 (Lemma 6.6 in Chen and Price (2019a)) . Let D ′ be an arbitrary distribution over G and let K IS ,D ′ be the condition number of importance sampling from D ′ (defined by Eq. (5) ). There exists an absolute constant C such that for any ε ∈ (0 , 1) and δ ∈ (0 , 1) , let S = { t 1 , . . . , t s } be a set of i.i.d. samples from the distribution D ′ and let w be the weight vector defined by w j = D ( t j ) s · D ′ ( t j ) for each j ∈ [ s ] . Then, as long as the s × d matrix A i,j = √ w i · v j ( t i ) satisfies ∥ A ∗ A -I ∥ 2 ≤ ε with probability at least 1 -δ.

<!-- formula-not-decoded -->

## C Energy Bounds for Band-limited Signals

The energy bound shows that the maximum value of a band-limited signal in a certain interval can be bounded by its energy on the interval. One interesting fact is that the approximation ratio in the energy bound is only relate to the sparsity k , and have no relationship with time duration T and band-limit F . An application of energy bound is preserving the norm, that is what is the least size of set S , such that ∥ f ∥ S = ∥ f ∥ T , for any function f in a certain function family. The relationship between energy bound and norm preserving can be build by Chernoff bound.

Borwein and Erdélyi (2006); Kós (2008); Chen et al. (2016); Chen and Price (2019b) proved energy bounds for sparse Fourier signal under one-dimensional continuous Fourier transform. We further generalize these results to discrete band-limited signal under discrete Fourier transform and highdimensional band-limited signal under continuous Fourier transform.

This section is organized as follows:

- Section C.1 reviews previous results for one-dimensional continuous Fourier-sparse signals.
- Section C.2 builds the connection between energy bound and the concentration property.

## C.1 Energy bound for one-dimensional signals

In this section, we review the energy bound proved in prior work Borwein and Erdélyi (2006); Kós (2008); Chen et al. (2016); Chen and Price (2019b).

Kós (2008) proved the following energy bound:

Theorem C.1 (Kós (2008); Chen et al. (2016)) . Define a family of F -band-limit, k -sparse Fourier signals:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, for any t ∈ ( -1 , 1) ,

Borwein and Erdélyi (2006) also proved a time-dependent energy bound for one-dimensional signal: Theorem C.2 (Borwein and Erdélyi (2006); Chen and Price (2019a)) . Define a family of F -band-limit, k -sparse Fourier signals:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, for any t ∈ ( -1 , 1) ,

## C.2 Energy bounds imply concentrations

By using Chernoff bound, we prove the following lemma to show the performance of uniformly sampling.

## C.2.1 Continuous case

Lemma C.3. Let d ∈ Z + . Let R be a parameter. Given any function x ( t ) : R d → C with max t ∈ [0 ,T ] d | x ( t ) | 2 ≤ R ∥ x ( t ) ∥ 2 T . Let S denote a set of points chosen uniformly at random from [0 , T ] d .

<!-- formula-not-decoded -->

We have that

where ∥ x ( t ) ∥ 2 T = 1 T d ∫ [0 ,T ] d | x ( t ) | 2 d t .

Proof. Let M denote max t ∈ [0 ,T ] d | x ( t ) | 2 . Replacing X i by | x ( t i ) | 2 M and n by | S | in Lemma A.1, we obtain that

<!-- formula-not-decoded -->

The above equation implies

Multiplying M on the both sides

<!-- formula-not-decoded -->

Applying bound on µ

<!-- formula-not-decoded -->

which is less than 2 exp( -ε 2 3 | S | /R )

<!-- formula-not-decoded -->

## D Uniform Sketching Band-Limited Signals

In this section, we show an intermediate step in the reduction from Frequency estimation to Signal estimation: constructing a small sketching subset S of the time domain obliviously (without making any query to the signal), so that the signal discretized by S has norm close to the original continuous signal. More formally, we define the uniform sketching Fourier signal problem as follows:

̸

Let ε ∈ (0 , 0 . 1) denote the accuracy parameter. Find a set S = { t 1 , . . . , t s } ⊆ [0 , T ] d of size s such that

Problem D.1 (Uniform sketching band-limited signal problem) . Suppose f 1 , f 2 , · · · , f k ∈ R d , and v 1 , . . . , v k ∈ C . Define the continuous signal x ( t ) = ∑ k j =1 v j e 2 π i ⟨ f j ,t ⟩ . Let η = min i = j ∥ f j -f i ∥ ∞ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

In Section D.1, we show how to sketch one-dimensional signals with nearly-optimal weighted sketching.

## D.1 Weighted uniform sketching one-dimensional signals

For one-dimensional signals, the most natural approach to uniform sketching is to uniformly sample some points in the time domain. However, by a standard concentration argument, we know that the sample complexity is poly( k ) , which is not time-efficient for our task. In this section, we show a more efficient sketching method for one-dimensional band-limited signals by assigning different weights to each sample point. More precisely, let S = { t 1 , . . . , t s } ⊆ [0 , T ] be a discrete sketching set and let w ∈ R s ≥ 0 be the weight vector. We define the weighted sketching norm of the signal as follows:

<!-- formula-not-decoded -->

And the goal of weighted uniform sketching is to find a small set S and a weight vector w such that ∥ x ∥ S,w ≈ ∥ x ∥ T .

In the following lemma, we give a sketch for any one-dimensional band-limited signal with nearlyoptimal size:

Lemma D.2 (Nearly-optimal weighted sketch for one-dimensional signals) . For k ∈ N + , define a probability distribution D ( t ) as follows:

where c = Θ( T -1 log -1 ( k )) is a normalization factor such that ∫ T -T D ( t )d t = 1 .

<!-- formula-not-decoded -->

For any f 1 , · · · , f k ∈ [ -F, F ] and v 1 , · · · , v k ∈ C , let the continuous signal x ( t ) = ∑ k j =1 v j exp(2 π i f j t ) . For any ε, ρ ∈ (0 , 1) , let S D = { t 1 , · · · , t s } be a set of i.i.d. samples from D ( t ) of size s ≥ O ( ε -2 k log( k ) log(1 /ρ )) . Let the weight vector w ∈ R s be defined by w i := 2 / ( TsD ( t i )) for i ∈ [ s ] . Then with probability at least 1 -ρ , we have where ∥ x ∥ 2 T := 1 2 T ∫ T -T | x ( t ) | 2 d t .

<!-- formula-not-decoded -->

Proof. For the convenient, in the proof, we use time duration [ -T, T ] . Let F be defined as:

Let { v 1 ( t ) , v 2 ( t ) , · · · , v k ( t ) } be an orthonormal basis for F with respect to the distribution D , i.e.,

<!-- formula-not-decoded -->

We first prove that the distribution D is well-defined. By the condition that ∫ T -T D ( t )d t = 1 , we have which implies that

<!-- formula-not-decoded -->

Thus, we get that c = Θ( T -1 log -1 ( k )) .

To show that sampling from distribution D give a good weighted sketch, we will use some technical tools in Section B.3. Applying Lemma B.12 with D ′ = D , D = Uniform([ -T, T ]) , d = k , δ = ρ , we have that, with probability at least 1 -ρ , the matrix A ∈ C s × k defined by A i,j := √ w i · v j ( x i ) satisfying

<!-- formula-not-decoded -->

as long as s ≥ C ε 2 · K IS ,D ′ log k ρ . Then, by Lemma B.11, it implies that for every x ∈ F ,

<!-- formula-not-decoded -->

It remains to bound the size of S D ; or equivalently, we need to upper-bound the condition number of the importance sampling of D ′ (see Definition B.10):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover,

Hence, we conclude that,

<!-- formula-not-decoded -->

where the first step follows from the definition, the second step follows from D ( t ) = Uniform([ -T, T ])( t ) = 1 2 T , the third step follows from Theorem C.1 and Theorem C.2, and the remaining steps follow from direct calculations. Thus, we get that

The lemma is then proved.

<!-- formula-not-decoded -->

## D.2 ε -net for sparse band-limited signals

In this section, we construct ε -nets for high-dimensional sparse Fourier continuous and discrete signals.

Lemma D.3 ( ε -net construction for continuous signals) . Given k ∈ Z + unknown frequencies f 1 , f 2 , . . . , f k ∈ [ -F, F ] d . Let V := { e 2 π i ⟨ f i ,t ⟩ | i ∈ [ k ] } be a family of Fourier basis. Let Q := { u ∈ span { V } | ∥ u ∥ 2 T = 1 } be the set of all signals in [0 , T ] d with frequency f 1 , . . . , f k , where ∥ x ∥ 2 T = 1 T d ∫ [0 ,T ] d | x ( t ) | 2 d t .

Then, there exists an ε -net P d ⊂ Q such that

1. ∀ u ∈ Q , ∃ w ∈ P d , ∥ u -w ∥ T ≤ ε.

Proof. We first construct an ε k -net for the unit disk in C , i.e., { z ∈ C | | z | ≤ 1 } . Let P ′ denote

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that | ε/ (2 k ) j 1 | ≤ ε/ (2 k ) · 2 k/ε = 1 ; and similarly, | ε/ (2 k ) j 2 | ≤ 1 . Thus, for any a ∈ C , | a | ≤ 1 , there is a b ∈ P ′ such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- P ′ is an ε k -net in the unit circle of C .
- P ′ has size at most (4 k ε +1) 2 .

Then, we use P ′ to construct an ε -net for Q . Since the dimension of Q is at most k , we take an orthonormal basis w 1 , · · · , w k ∈ Q such that,

<!-- formula-not-decoded -->

And we define

First, for any u ∈ Q , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies that | α ′ i | ≤ 1 for all i ∈ [ k ] . So, for any a ∈ Q , there is a b ∈ P ′′ such that ∥ a -b ∥ T ≤ k · ε/k = ε . Moreover, |P ′′ | ≤ ((4 k ε +1) 2 ) k ≤ (5 k ε ) 2 k . Therefore, we conclude that P ′′ is an ε -net for Q and |P ′′ | ≤ ( 5 k ε ) 2 k . Then we define therefore we have that, for any a ∈ Q , there is a b ∈ P ′′ such that ∥ a -b ∥ T ≤ ε , because there is a c ∈ P d , such that ∥ c -b ∥ T = min d ∈Q ∥ d -b ∥ T ≤ ∥ a -b ∥ T ≤ ε . Then, ∥ c -a ∥ T ≤ ∥ c -b ∥ T + ∥ b -a ∥ T ≤ 2 ε .

## E Fast Implementation of Well-Balanced Sampling Procedure

Well-balanced sampling procedure was first defined in Chen and Price (2019a) to study the active linear regression problem. Our signal estimation algorithm will call it as a sub-procedure. In this section, we give a fast implementation of well-balanced sampling procedure based on the Randomized BSS algorithm Batson et al. (2012); Lee and Sun (2015).

First, we restate the definition of well-balanced sampling procedure in Chen and Price (2019a). Definition E.1 (Well-balanced sampling procedure (WBSP), Chen and Price (2019a)) . Given a linear family F and underlying distribution D , let P be a random sampling procedure that terminates in m iterations ( m is not necessarily fixed) and provides a coefficient α i and a distribution D i to sample x i ∼ D i in every iteration i ∈ [ m ] .

We say P is an ε -WBSP if it satisfies the following two properties:

<!-- formula-not-decoded -->

This definition describes a general sampling procedure that uses a few samples to represent the whole continuous signal, and the sampling procedure should satisfy two properties: one guarantees that the norm of any function in a function family is preserved, and another guarantees that the norm of noise is also preserved.

<!-- formula-not-decoded -->

In Section E.1, we review some results in Chen and Price (2019a) and show that WBSP can be implemented via randomized spectral sparsification. In Section E.2, we design a data structure and improve the time efficiency of the WBSP. In Section E.3, we discover a tradeoff between the preprocessing cost and the query cost, which can improve the space complexity.

## E.1 Randomized BSS implies a WBSP

In this section, we review the result of Chen and Price (2019a), which shows that the Randomized BSS algorithm Batson et al. (2012); Lee and Sun (2015) implies a well-balanced sampling procedure. Lemma E.2 (Lemma 5.1 in Chen and Price (2019a)) . Let G be any domain. Given any dimension d linear function family of function f : G C ,

<!-- formula-not-decoded -->

F →

where u j : G → C . Given any distribution D over G , and any ε &gt; 0 , there exists an efficient procedure (Algorithm 2) that runs in O ( ε -1 d 3 | G | + ε -1 d ω +1 ) time and outputs a set S ⊆ G and weight w such that

- | S | = O ( d/ε ) , w ∈ R | S | ,
- the procedure is an ε -WBSP,

holds with probability 1 -1 200 .

Algorithm 2 A well-balanced sampling procedure based on Randomized BSS (see Chen and Price (2019a))

<!-- formula-not-decoded -->

## E.2 Fast implementation of WBSP

In this section, we give a fast implementation of Algorithm 2:

Theorem E.3 (Fast implementation of WBSP) . Let G be any domain. Given any dimension d linear function family F of function f : G → C ,

<!-- formula-not-decoded -->

where u j : G → C . Given any distribution D over G , and any ε &gt; 0 , there exists an efficient procedure (Algorithm 3) that runs in O ( d 2 | G | + ε -1 d 3 log | G | + ε -1 d ω +1 ) time and outputs a set S ⊆ G and weight w ∈ R | S | such that the following properties hold with probability at least 0.995:

- | S | = O ( d/ε ) ,
- the procedure is an ε -WBSP.

Our algorithm is based on a data structure for solving the online quadratic-form sampling problem defined as follows:

- .

## Algorithm 3 Our fast implementation of well-balanced sampling procedure

```
1: procedure RANDBSS+( d, F , D, ε ) ▷ Theorem E.3 2: /*Preprocessing*/ 3: Find an orthonormal basis v 1 , . . . , v d of F under D 4: γ ← √ ε/ 3 and mid ← 4 d/γ 1 / (1 -γ ) -1 / (1+ γ ) 5: j ← 0 , B 0 ← 0 6: l 0 ←-2 d/γ, u 0 ← 2 d/γ 7: δ ← 1 / poly( d ) 8: ▷ Let v ( x ) = ( v 1 ( x ) , . . . , v d ( x ) ) ∈ R d 9: DS . INIT ( | D | , d, { v ( x 1 ) , · · · , v ( x | D | ) } ⊂ R d , { D ( x 1 ) , . . . , D ( x | D | ) } ⊂ R ) ▷ Algorithm 4 10: /*Iterative step*/ 11: while u j +1 -l j +1 < 8 d/γ do 12: Φ j ← tr[( u j I -B j ) -1 ] + tr[( B j -l j I ) -1 ] ▷ The potential function at iteration j . 13: α j ← γ Φ j · 1 mid 14: E j ← ( u j I -B j ) -1 +( B j -l j I ) -1 15: q ← DS . QUERY ( E j / Φ j ) ▷ q ∈ [ | D | ] , Algorithm 4 16: x j ← x q and set a scale s j ← γ v ( x j ) ⊤ E j v ( x j ) 17: B j +1 ← B j + s j · v ( x j ) v ( x j ) ⊤ 18: u j +1 ← u j + γ Φ j (1 -γ ) , l j +1 ← l j + γ Φ j (1+ γ ) 19: j ← j +1 20: end while 21: m ← j 22: Assign the weight w j ← s j / mid for each x j 23: return { x 1 , x 2 , · · · , x m } , w 24: end procedure
```

Problem E.4 (Online Quadratic-Form Sampling Problem) . Given n vectors v 1 , . . . , v n ∈ R d and n coefficients α 1 , . . . , α n , for any PSD matrix A ∈ R d × d , output a sample i ∈ [ n ] from the following distribution D A :

Theorem E.5. There is a data structure (Algorithm 4) that uses O ( nd 2 ) spaces for the Online Quadratic-Form Sampling Problem with the following procedures:

<!-- formula-not-decoded -->

- INIT ( n, d, { v 1 , . . . , v n } ⊂ R d , { α 1 , . . . , α n } ⊂ R ) : the data structure preprocesses in time O ( nd 2 ) .
- QUERY ( A ∈ R d × d ) : Given a PSD matrix A , the QUERY operation samples i ∈ [ n ] exactly from the probability distribution D A defined in Problem E.4 in O ( d 2 log n ) -time.

Proof. The pseudo-code of the algorithm is given as Algorithm 4. The idea is to build a binary tree such that each node has an interval in [ l, . . . , r ] ⊂ [1 , . . . , n ] and stores a matrix ∑ r i = l α i · v i v ⊤ i . For each internal node with interval [ l, . . . , r ] , its left child node has interval [ l, . . . , ⌊ ( l + r ) / 2 ⌋ ] , and its right child node has interval [ ⌊ ( l + r ) / 2 ⌋ +1 , . . . , r ] .

<!-- formula-not-decoded -->

We first prove the correctness. Suppose the output of QUERY is i ∈ [ n ] . We compute its probability. Let u 0 = root , u 1 , . . . , u t be the path from the root of the tree to the leaf with id = i . Then, we have where [ l j , . . . , r j ] is the range of the node u j , the first step follows from the conditional probability, the second step follows from Line 34 in Algorithm 4, and the last step follows from the telescoping products. Hence, we get that

<!-- formula-not-decoded -->

Hence, the sampling distribution is the same as the Online Quadratic-Form Sampling Problem's distribution.

For the running time, in the preprocessing stage, we build the binary tree recursively. It is easy to see that the number of nodes in the tree is O ( n ) and the depth is O (log n ) . For a leaf node, we take O ( d 2 ) -time to compute the matrix α i · v i v ⊤ i ∈ R d × d . For an internal node, we take O ( d 2 ) -time to add up the matrices of its left and right children. Thus, the total preprocessing time is O ( nd 2 ) .

In the query stage, we walk along a path from the root to a leaf, which has O (log n ) steps. In each step, we compute the inner product between A and the current node's matrix, which takes O ( d 2 ) -time. And we compute the inner product between A and its left child node's matrix, which also takes O ( d 2 ) -time. Then, we toss a coin and decide which subtree to move. Hence, each query takes O ( d 2 log n ) -time.

The theorem is then proved.

Lemma E.6 (Running time of Procedure RANDBSS+ in Algorithm 3) . Algorithm 3 runs in

- O ( | D | d 2 ) -time for preprocessing,
- O ( ε -1 d ) iterations.
- O ( d 2 log( | D | ) + d ω ) -time per iteration, and

Thus, the total running time is,

<!-- formula-not-decoded -->

Proof. In each call of the Procedure RANDBSS+ in Algorithm 3,

- Finding orthonormal basis takes O ( | D | d 2 ) .
- In the line 9, it runs O ( | D | d 2 ) times.
- The while loop repeat O ( ε -1 d ) times.
- -Line 14 is computing ( u j I -B j ) ∈ C d × d , ( u j I -B j ) -1 . This part takes O ( d ω ) time 6 .
- -Note that line 15 of Procedure RANDBSS+ in Algorithm 3 runs O ( d 2 log | D | ) times.

So, the time complexity of Procedure RANDBSS+ in Algorithm 3 is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma E.7 (Correctness of Procedure RANDBSS+ in Algorithm 3) . Given any dimension d linear space F , any distribution D over the domain of F , and any ε &gt; 0 , RANDBSS+ ( d, F , D, ε ) is an ε -WBSP that terminates in O ( d/ε ) rounds with probability 1 -1 / 200 .

Proof. We first claim that, for each j ∈ [ m ] , x j has the same distribution as D j , where

Notice that sampling from distribution D j can be reformulated as an Online Quadratic-Form Sampling Problem: the vectors are { v ( x ) } x ∈ D , the coefficients are { D ( x ) } x ∈ D , and the query matrix is E ′ j := E j / Φ j . Then, we have D j = D E ′ j defined in Problem E.4. Hence, by Theorem E.5, we can use the data structure (Algorithm 4) to efficiently sample from D j .

<!-- formula-not-decoded -->

Therefore, the sample x j in each iteration is generated from the same distribution as the original randomized BSS algorithm (Algorithm 2). Then, the WBSP guarantee and the number of iterations immediately follow from the proof of (Chen and Price, 2019a, Lemma 5.1).

The proof of the lemma is then completed.

6 Note that this step seems to be very difficult to speed up via the Sherman-Morrison formula since u j changes in each iteration and the update is of high rank.

## Algorithm 4 Quadratic-form sampling data structure

̸

```
1: structure Node 2: V ∈ R d × d 3: left , right ▷ Point to the left/right child in the tree 4: end structure 5: data structure DS 6: members 7: n ∈ N ▷ The number of vectors 8: v 1 , . . . , v n ∈ R d ▷ d -dimensional vectors 9: α 1 , . . . , α n ∈ R ▷ Coefficients 10: root : Node ▷ The root of the tree 11: end members 12: procedure BUILDTREE( l, r ) ▷ [ l, . . . , r ] is the range of the current node 13: p ← new Node 14: if l = r then ▷ Leaf node 15: p .V ← α l · v l v ⊤ l ▷ It takes O ( d 2 ) -time 16: else ▷ Internal node 17: mid ←⌊ ( l + r ) / 2 ⌋ 18: p . left ← BUILDTREE ( l, mid ) 19: p . right ← BUILDTREE ( mid +1 , r ) 20: p .V ← ( p . left) .V +( p . right) .V ▷ It takes O ( d 2 ) -time 21: end if 22: return p 23: end procedure 24: procedure INIT( n, d, { v i } i ∈ [ n ] ⊆ R d , { α i } i ∈ [ n ] ⊆ R ) 25: v i ← v i , α i ← α i for i ∈ [ n ] 26: root ← BUILDTREE (1 , n ) 27: end procedure 28: procedure QUERY( A ∈ R d × d ) 29: p ← root , l ← 1 , r ← n 30: s ← 0 31: while l = r do ▷ There are O (log n ) iterations 32: w ←⟨ p .V, A ⟩ ▷ It takes O ( d 2 ) -time 33: w ℓ ←⟨ ( p . left) .V, A ⟩ 34: Sample c from Bernoulli( w ℓ /w ) 35: if c = 0 then 36: p ← p . left , r ←⌊ ( l + r ) / 2 ⌋ 37: else 38: p ← p . right , l ←⌊ ( l + r ) / 2 ⌋ +1 39: end if 40: end while 41: return l 42: end procedure 43: end data structure
```

Proof of Theorem E.3. The running time of the algorithm follows from Lemma E.6, and the correctness follows from Lemma E.7.

## E.3 Trade-off between preprocessing and query

In this section, we consider the preprocessing and query trade-off in the data structure for quadratic form sampling problem. In the following theorem, we give a new data structure that takes less time in preprocessing and more time for each query than Theorem E.5, and the space complexity is also reduced from O ( nd 2 ) to O ( nd ) .

Theorem E.8. There is a data structure (Algorithms 5 and 6) that uses O ( nd ) spaces for the Online Quadratic-Form Sampling Problem with the following procedures:

- INIT ( n, d, { v 1 , . . . , v n } ⊂ R d , { α 1 , . . . , α n } ⊂ R ) : the data structure preprocesses in time O ( nd ω -1 ) .
- QUERY ( A ∈ R d × d ) : Given a PSD matrix A , the QUERY operation samples i ∈ [ n ] exactly from the probability distribution D A defined in Problem E.4 in O ( d 2 log( n/d ) + d ω ) -time.

Proof. The time and space complexities follow from Lemma E.9. And the correctness follows from Lemma E.10.

```
1: structure Node 2: V 1 , V 2 ∈ R d × d 3: left , right ▷ Point to the left/right child in the tree 4: end structure 5: data structure DS+ ▷ Theorem E.8 6: members 7: n ∈ N ▷ The number of vectors 8: m ∈ N ▷ The number of blocks 9: v 1 , . . . , v n ∈ R d ▷ d -dimensional vectors 10: root : Node ▷ The root of the tree 11: end members 12: procedure BUILDTREE( l, r ) ▷ [ l, . . . , r ] is the range of the current node 13: p ← new Node 14: if l = r then ▷ Leaf node 15: p .V 2 ← [ v ( l -1) d +1 · · · v ld ] 16: p .V 1 ← ( p .V 2 ) · ( p .V 2 ) ⊤ ▷ It takes O ( d ω ) -time 17: ▷ p . mat1 = ∑ ld i =( l -1) d +1 v i v ⊤ i 18: else ▷ Internal node 19: mid ←⌊ ( l + r ) / 2 ⌋ 20: p . left ← BUILDTREE ( l, mid ) 21: p . right ← BUILDTREE ( mid +1 , r ) 22: p .V 1 ← ( p . left) .V 1 +( p . right) .V 1 ▷ It takes O ( d 2 ) -time 23: end if 24: return p 25: end procedure 26: procedure INIT( n, d, { v i } i ∈ [ n ] ⊆ R d , { α i } i ∈ [ n ] ⊆ R ) 27: v i ← v i · √ α i for i ∈ [ n ] 28: m ← n/d ▷ We assume that n is divisible by d 29: Group { v i } i ∈ [ n ] into m blocks B 1 , . . . , B m ▷ B i = { v ( i -1) d +1 , . . . , v id } for i ∈ [ m ] 30: root ← BUILDTREE (1 , m ) 31: end procedure 32: end data structure
```

Algorithm 5 Quadratic-form sampling with preprocessing-query trade-off: Preprocessing

Lemma E.9 (Time and space complexities of Algorithms 5 and 6) . The INIT procedure takes O ( nd ω -1 ) -time. The QUERY procedure takes O ( d 2 log( n/d ) + d ω ) -time. The data structure uses O ( nd ) -space.

Proof. We prove the space and time complexities of the data structure as follows: Space complexity: Let m = n/d . It is easy to see that there are O ( m ) nodes in the data structure. And each node has two d -byd matrices. Hence, the total space used by the data structure is O ( n/d ) · O ( d 2 ) = O ( nd ) .

Time complexity: In the preprocessing stage, the time-consuming step is the call of BUILDTREE. There are O ( m ) internal nodes and O ( m ) leaf nodes. Each internal node takes O ( d 2 ) -time to construct the matrix V 1 (Line 22). For each leaf node, it takes O ( d 2 ) -time to form the matrix V 2 (Line 15). And it takes O ( d ω ) -time to compute the matrix V 1 (Line 16). Hence, the total running time of BUILDTREE is O ( md ω ) = O ( nd ω -1 ) .

Algorithm 6 Quadratic-form sampling with preprocessing-query trade-off: Query

̸

```
1: data structure DS+ ▷ Theorem E.8 2: members 3: n ∈ N ▷ The number of vectors 4: m ∈ N ▷ The number of blocks 5: v 1 , . . . , v n ∈ R d ▷ d -dimensional vectors 6: root : Node ▷ The root of the tree 7: end members 8: procedure BLOCKSAMPLING( p , l ∈ N , A ∈ R d × d ) ▷ p is a leaf node with index l 9: U ← ( p .V 2 ) ⊤ · A · ( p .V 2 ) ▷ It takes O ( d ω ) -time 10: Define a distribution D l over [ d ] such that D l ( i ) ∝ U i,i 11: Sample i ∈ [ d ] from D l ▷ It takes O ( d ) -time 12: return ( l -1) d + i 13: end procedure 14: procedure QUERY( A ∈ R d × d ) 15: p ← root , l ← 1 , r ← m 16: s ← 0 17: while l = r do ▷ There are O (log m ) iterations 18: w ←⟨ p .V 1 , A ⟩ ▷ It takes O ( d 2 ) -time 19: w ℓ ←⟨ ( p . left) .V 1 , A ⟩ 20: Sample c from Bernoulli( w ℓ /w ) 21: if c = 0 then 22: p ← p . left , r ←⌊ ( l + r ) / 2 ⌋ 23: else 24: p ← p . right , l ←⌊ ( l + r ) / 2 ⌋ +1 25: end if 26: end while 27: return BLOCKSAMPLING( p , l , A ) 28: end procedure 29: end data structure
```

In the query stage, the While loop in the QUERY procedure (Line 17) is the same as in Algorithm 4. Since there are O ( m ) nodes in the tree, it takes O ( d 2 log m ) -time. Then, in the BLOCKSAMPLING procedure, it takes O ( d ω ) -time to compute the matrix U (Line 9), and it takes O ( d ) -time to sample an index from the distribution D l (Line 11). Hence, the total running time for each query is O ( d 2 log m + d ω ) = O ( d 2 log( n/d ) + d ω ) .

<!-- formula-not-decoded -->

Lemma E.10 (Correctness of Algorithm 6) . The distribution of the output of the QUERY ( A ) is D A defined by Eq. (7) .

Proof. For simplicity, we assume that all the coefficients α i = 1 .

Let u 0 = root , u 1 , . . . , u t be the path in the While loop (Line 17) from the root of the tree to the leaf with index l ∈ [ m ] . By the construction of leaf node, we have which is the same as the V -matrix in Algorithm 4. Hence, similar to the proof of Theorem E.5, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where { ( l -1) d +1 , . . . , ld } is the range of the node u t and { 1 , . . . , n } is the range of u 0 .

Then, consider the BLOCKSAMPLING procedure. Let { v 1 , . . . , v d } be the vectors in the input block. At Line 9, we have

For i ∈ [ d ] , the i -th element in the diagonal of U is

Hence,

<!-- formula-not-decoded -->

Therefore, for any k ∈ [ n ] , if k = ( l -1) d + r for some l, r ∈ N , then the sample probability is

<!-- formula-not-decoded -->

The lemma is then proved.

As a corollary, we get a WBSP using less space:

Corollary E.11 (Space efficient implementation of WBSP) . By plugging-in the new data structure (Algorithms 5 and 6) to FASTERRANDSAMPLINGBSS (Algorithm 3), we get an algorithm taking O ( | D | d 2 + γ -2 d · ( d 2 log | D | + d ω )) -time and using O ( | D | d ) -space.

Proof. In the preprocessing stage of FASTERRANDSAMPLINGBSS, we take O ( | D | d 2 ) -time for Gram-Schmidt process and O ( | D | d ω -1 ) -time for initializing the data structure (Algorithm 5).

The number of iterations is γ -2 d . In each iteration, the matrix E j can be computed in O ( d ω ) -time. And querying the data structure takes O ( d 2 log( | D | /d ) + d ω ) -time.

Hence, the total running time is

For the space complexity, the data structure uses O ( | D | d ) -space. The algorithm uses O ( d 2 ) extra space in preprocessing and each iteration. Hence, the total space complexity is O ( | D | d ) .

<!-- formula-not-decoded -->

## F Sketch Distillation for Fourier Sparse Signals

In Section D, we show an oblivious approach for sketching Fourier sparse signals. However, there are two issues of using this sketching method in Signal estimation: 1. The sketch size too large. 2. The noise in the observed signal could have much larger energy on the sketching set than its average energy. To resolve these two issues, in this section, we propose a method called sketch distillation to post-process the sketch obtained in Section D that can reduce the sketch size to O ( k ) and prevent the energy of noise being amplified too much. However, we need some extra information about the signal x ∗ ( t ) : we assume that the frequencies of the noiseless signal x ( t ) are known. But the sketch distillation process can still be done partially oblivious , i.e., we do not need to access/sample the signal.

In Section F.1, we show our distillation algorithms for one-dimensional signals.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F.1 Sketch distillation for one-dimensional signals

In this section, we show how to distill the sketch produced by Lemma D.2 from O ( k log k ) -size to O ( k ) -size, using an ε -well-balanced sampling procedure developed in Section E.

̸

Lemma F.1 (Fast distillation for one-dimensional signal) . Given f 1 , f 2 , · · · , f k ∈ R . Let x ∗ ( t ) = ∑ k j =1 v j exp(2 π i f j t ) . Let η = min i = j | f j -f i | . For any accuracy parameter ε ∈ (0 , 0 . 1) , there is an algorithm FASTDISTILL1D (Algorithm 7) that runs in O ( ε -2 k ω +1 ) -time and outputs a set S ⊂ [ -T, T ] of size s = O ( k/ε 2 ) and a weight vector w ∈ R s ≥ 0 such that,

<!-- formula-not-decoded -->

holds with probability 0 . 99 .

Furthermore, for any noise signal g ( t ) , the following holds with high probability:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. For the convenient, in the proof, we use time duration [ -T, T ] . Let D ( t ) be defined as follows:

where c = O ( T -1 log -1 ( k )) a fixed value such that ∫ T -T D ( t )d t = 1 .

<!-- formula-not-decoded -->

First, we randomly pick up a set S 0 = { t 1 , · · · , t s 0 } of s 0 = O ( ε -2 0 k log( k ) log(1 /ρ 0 )) i.i.d. samples from D ( t ) , and let w ′ i := 2 / ( Ts 0 D ( t i )) for i ∈ [ s 0 ] be the weight vector, where ε 0 , ρ 0 are parameters to be chosen later.

By Lemma D.2, we know that ( S 0 , w ′ ) gives a good weighted sketch of the signal that can preserve the norm with high probability. More specifically, with probability 1 -ρ 0 ,

<!-- formula-not-decoded -->

Then, we will select s = O ( k/ε 2 1 ) elements from S 0 and output the corresponding weights w 1 , w 2 , · · · , w s by applying RANDBSS+ with the following parameter: replacing d by k , ε by ε 2 1 , and D by D ( t i ) = w ′ i / ∑ j ∈ [ s 0 ] w ′ j for i ∈ [ s 0 ] . By Theorem E.3 and the property of WBSP (Definition E.1), we obtain that with probability 0 . 995 ,

<!-- formula-not-decoded -->

Combining with Eq. (8), we conclude that

<!-- formula-not-decoded -->

where the second step follows from Eq. (8) and the last stpe follows by taking ε 0 = ε 1 = ε/ 4 .

The overall success probability follows by taking union bound over the two steps and taking ρ 0 = 0 . 001 . The running time of Algorithm 7 follows from Claim F.2. And the furthermore part follows from Claim F.3.

The proof of the lemma is then completed.

Claim F.2 (Running time of Procedure FASTDISTILL1D in Algorithm 7) . Procedure FASTDISTILL1D in Algorithm 7 runs in time.

<!-- formula-not-decoded -->

## Algorithm 7 Fast distillation for one-dimensional signal

- 1: procedure WEIGHTEDSKETCH( k, ε, T, B )
- 3: D ( t ) is defined as follows:
- 2: c ← O ( T -1 log -1 ( k ))

<!-- formula-not-decoded -->

- 4: S 0 ← O ( ε -2 k log( k )) i.i.d. samples from D
- 6: w t ← 2 T ·| S 0 |· D ( t )
- 5: for t ∈ S 0 do
- 7: end for
- 10: end procedure
- 8: Set a new distribution D ′ ( t ) ← w t / ∑ t ′ ∈ S 0 w t ′ for all t ∈ S 0 9: return D ′
- 11: procedure FASTDISTILL1D( k , ε , F = { f 1 , . . . , f k } , T )
- 13: Set the function family F as follows:
- 12: Distribution D ′ ← WEIGHTEDSKETCH ( k, ε, T, B )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 16: end procedure

Proof. First, it is easy to see that Procedure WEIGHTEDSKETCH takes O ( ε -2 k log( k )) -time.

By Theorem E.3 with | D | = O ( ε -2 k log( k )) , d = k , we have that the running time of Procedure RANDBSS+ is

Hence, the total running time of Algorithm 7 is O ( ε -2 k ω +1 ) .

<!-- formula-not-decoded -->

Claim F.3 (Preserve the energy of noise) . Let ( S, w ) be the outputs of Algorithm 7. Then, we have that

<!-- formula-not-decoded -->

holds with probability 0 . 99 .

Proof. For the convenient, in the proof, we use time duration [ -T, T ] . Algorithm 7 has two stages of sampling.

In the first stage, Procedure WEIGHTEDSKETCH samples a set S 0 = { t ′ 1 , . . . , t ′ s 0 } of i.i.d. samples from the distribution D , and a weight vector w ′ . Then, we have

<!-- formula-not-decoded -->

▷ Lemma D.2

▷

Lemma F.1

<!-- formula-not-decoded -->

where the first step follows from the definition of the norm, the third step follows from the definition of w i , the forth step follows from E t ∼ D 0 ( t ) [ D 1 ( t ) D 0 ( t ) f ( t )] = E t ∼ D 1 ( t ) f ( t ) .

In the second stage, let P denote the Procedure RANDBSS+. With high probability, P is a ε -WBSP (Definition E.1). By the Definition E.1, each sample t i ∼ D i ( t ) and w i = α i · D ′ ( t i ) D i ( t i ) in every iteration i ∈ [ s ] , where ∑ s i =1 α i ≤ 5 / 4 and D ′ ( t ) = w ′ t ∑ t ′ ∈ S 0 w ′ t ′ . As a result,

<!-- formula-not-decoded -->

where the first step follows from the definition of the norm, the third step follows from w i = α i · D ′ ( t i ) D i ( t i ) , the forth step follows from E t ∼ D 0 ( t ) D 1 ( t ) D 0 ( t ) f ( t ) = E t ∼ D 1 ( t ) f ( t ) , the sixth step follows from D ′ ( t ) = w ′ t ∑ t ′ ∈ S 0 w ′ t ′ and the definition of the norm, the last step follows from ∑ s i =1 α i ≤ 5 / 4 and ( ∑ t ′ ∈ S 0 w ′ t ′ ) -1 = O ( ρ -1 ) with probability at least 1 -ρ/ 2 . Hence, combining the two stages together, we have

And by Markov inequality and union bound, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F.1.1 Sharper bound for the energy of orthogonal part of noise

In this section, we give a sharper analysis for the energy of g ⊥ on the sketch, which is the orthogonal projection of g to the space F . More specifically, we can decompose an arbitrary function g into g ∥ + g ⊥ , where g ∥ ∈ F and ∫ [0 ,T ] h ( t ) g ⊥ ( t )d t = 0 for all h ∈ F . The motivation of considering g ⊥

is that g ∥ is also a Fourier sparse signal and its energy will not be amplified in the Signal Estimation problem. And the nontrivial part is to avoid the blowup of the energy of g ⊥ , which is shown in the following lemma:

Lemma F.4 (Preserving the orthogonal energy) . Let F be an m -dimensional linear function family with an orthonormal basis { v 1 , . . . , v m } with respect to a distribution D . Let P be the ε -WBSP that

generate a sample set S = { t 1 , . . . , t s } and coefficients α ∈ R s &gt; 0 , where each t i is sampled from distribution D i for i ∈ [ s ] . Define the weight vector w ∈ R s be such that w i := α i D ( t i ) D i ( t i ) for i ∈ [ s ] .

For any noise function g ( t ) that is orthogonal to F with respect to D , the following property holds with probability 0.99:

<!-- formula-not-decoded -->

Remark F.5. We note that this lemma works for both continuous and discrete signals.

where ⟨ g, v ⟩ S,w := ∑ s j =1 w j v ( t j ) g ( t j ) .

Remark F.6. |⟨ g, v i ⟩ S,w | 2 corresponds to the energy of g on the sketch points in S . On the other hand, if we consider the energy on the whole time domain, we have ⟨ g, v i ⟩ = 0 for all i ∈ [ m ] . The above lemma indicates that this part of energy could be amplified by at most O ( ε ) , as long as the sketch comes from a WBSP.

Proof. We can upper-bound the expectation of ∑ m i =1 |⟨ g, v i ⟩ S,w | 2 as follows:

<!-- formula-not-decoded -->

where the first step follows from Fact F.7, the second step follows from the definition of D ′ , the third follows from the linearity of expectation, the forth step follows from Fact F.8, the last step follows by pulling out the maximum value of w j ∑ k i =1 | v i ( t ) | 2 from the expectation. Next, we consider the first term:

<!-- formula-not-decoded -->

where the first step follows from the definition of w j , the second step follows from Fact F.9 that sup h ∈F { | h ( t j ) | 2 ∥ h ∥ 2 D } = ∑ k i =1 | v i ( t j ) | 2 , the last step follows from the definition of K IS ,D j (Eq. (5)).

Then, we bound the last term:

<!-- formula-not-decoded -->

Combining the two terms together, we have

<!-- formula-not-decoded -->

where the last step follows from P being a ε -WBSP(Definition E.1), which implies that ∑ s j =1 α j = 5 4 and α j K IS ,D j ≤ ε/ 2 for all j ∈ [ s ] .

Finally, by Markov's inequality, we have that holds with probability 0 . 99 .

Fact F.7.

where D ′ is a distribution defined by D ′ ( t i ) := w i ∥ w ∥ 1 for i ∈ [ s ] .

Proof. We have:

<!-- formula-not-decoded -->

Fact F.8. For any i ∈ [ m ] , we have

<!-- formula-not-decoded -->

Proof. We first show that for any i ∈ [ m ] and j ∈ [ s ] ,

<!-- formula-not-decoded -->

where the first step follows from the definition of w i , the third step follows from g ( t ) is orthonormal with v i ( t ) for any i ∈ [ k ] .

Then, we can expand LHS as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third step follows from the linearity of expectation, the fifth step follows from t j only depends on t 1 , . . . , t j -1 , and the sixth step follows from Eq. (9).

Fact F.9. Let { v 1 , . . . , v k } be an orthonormal basis of F with respect to the distribution D . Then, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Then, where the first step follows from each h ∈ F can be expanded as h = ∑ k i =1 a i v i and ∥ h ( t ) ∥ 2 D = ∥ a ∥ 2 2 (Fact B.9), the second step follows from the Cauchy-Schwartz inequality and taking a = v ( t ) ∥ v ( t ) ∥ 2 .

## G One-dimensional Signal Estimation

In this section, we apply the tools developed in previous sections to show two efficient reductions from Frequency Estimation to Signal Estimation for one-dimensional semi-continuous Fourier signals. The first reduction in Section G.1 is optimal in sample complexity, which takes linear number of samples from the signal but only achieves constant accuracy. The section reduction in Section G.2 takes nearly-linear number of samples but can achieve very high-accuracy (i.e., (1 + ε ) -estimation error).

## G.1 Sample-optimal reduction

The main theorem of this section is Theorem G.1. The optimal sample complexity is achieved via the sketch distillation in Lemma F.1.

Theorem G.1 (Sample-optimal algorithm for one-dimensional Signal Estimation) . For η ∈ R , let Λ( B ) ⊂ R denote the lattice Λ( B ) = { cη | c ∈ Z } . Suppose that f 1 , f 2 , · · · , f k ∈ Λ( B ) .

- 2: ε ← 0 . 01

Let x ∗ ( t ) = ∑ k j =1 v j exp(2 π i f j t ) , and let g ( t ) denote the noise. Given observations of the form x ( t ) = x ∗ ( t ) + g ( t ) , t ∈ [0 , T ] . Let η = min i = j | f j -f i | .

̸

Given D,η ∈ R + . Suppose that there is an algorithm FREQEST that

- takes S freq samples,
- runs in T freq -time, and
- outputs a set L of frequencies such that with probability 0 . 99 , the following condition holds:

<!-- formula-not-decoded -->

Then, there is an algorithm (Algorithm 8) such that

- takes O ( ˜ k + S freq ) samples · runs O ( k ω +1 + T freq ) time,

<!-- formula-not-decoded -->

- ˜ · outputs y ( t ) = ∑ ˜ k j =1 v ′ j · exp(2 π i f ′ j t ) with ˜ k = O ( |L| (1 + D/ ( Tη ))) such that with probability at least 0 . 9 , we have

## Algorithm 8 Signal estimation algorithm for one-dimensional signals (sample optimal version)

- 1: procedure SIGNALESTIMATIONFAST( x, k, F, T, B )
- 3: L ← FREQEST ( x, k, D, F, T, B )
- ˜ 5: s, { t 1 , t 2 , · · · , t s } , w ← FASTDISTILL1D ( ˜ k, √ ε, { f ′ i } i ∈ [ ˜ k ] , T, B ) ▷ ˜ k , w ∈ R ˜ k , Algorithm 7 6: A i,j ← exp(2 π i f ′ j t i ) , A ∈ C s × ˜ k

<!-- formula-not-decoded -->

- 7: b ← ( x ( t 1 ) , x ( t 2 ) , · · · , x ( t s )) ⊤
- 8: Solving the following weighted linear regression

<!-- formula-not-decoded -->

- 9: return y ( t ) = ∑ ˜ k j =1 v ′ j · exp(2 π i f ′ j t ) . end procedure

## 10:

Proof. First, we recover the frequencies by utilizing the algorithm FREQEST. Let L be the set of frequencies output by the algorithm FREQEST ( x, k, D, T, F, B ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We define ˜ L as follows:

We use ˜ k to denote the size of set ˜ L . And we use ˜ f 1 , ˜ f 2 , · · · , ˜ f ˜ k to denote the frequencies in the set ˜ L . It is easy to see that

Next, we focus on recovering magnitude v ′ ∈ C ˜ k . First we run Procedure FASTDISTILL1D in Algorithm 7 and obtain a set S = { t 1 , t 2 , · · · , t s } ⊂ [0 , T ] of size s = O ( ˜ k ) and a weight vector

▷ Fact A.4

▷

Theorem G.1

w ∈ R s &gt; 0 . Then, we sample the signal at t 1 , . . . , t s and let x ( t 1 ) , . . . , x ( t s ) be the samples. Consider the following weighted linear regression problem:

<!-- formula-not-decoded -->

where √ w := ( √ w 1 , . . . , √ w s ) , and the coefficients matrix A ∈ C s × ˜ k and the target vector b ∈ C s are defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, we output a signal where v ′ is an optimal solution of Eq. (10).

The running time follows from Lemma G.2. And the estimation error guarantee ∥ y ( t ) -x ( t ) ∥ T ≲ ∥ g ( t ) ∥ T follows from Lemma G.3.

The theorem is then proved.

Lemma G.2 (Running time of Algorithm 8) . Algorithm 8 takes O ( ˜ k ω +1 ) -time, giving the output of Procedure FREQEST .

Proof. At Line 5, we run Procedure FASTDISTILL1D, which takes O ( ˜ k ω +1 ) -time by Lemma F.1. At Line 8, we solve the weighted linear regression, which takes time by Fact A.4.

<!-- formula-not-decoded -->

Thus, the total running time is O ( ˜ k ω +1 ) . Lemma G.3 (Estimation error of Algorithm 8) . Let y ( t ) be the output signal of Algorithm 8. With high probability, we have

<!-- formula-not-decoded -->

Proof. We have

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows from Lemma F.1 with 0 . 99 probability, the third step follows from triangle inequality, the forth step follows from y ( t ) is the optimal solution of the linear system, the fifth step follows from Claim F.3, the sixth step follows from Lemma F.1, and the last step follows from the definition of g ( t ) .

## G.2 High-accuracy reduction

In this section, we prove Theorem G.4, which achieves (1 + ε ) -estimation error by a sharper bound on the energy of noise in Lemma F.4.

̸

Given D,η ∈ R + . Suppose that there is an algorithm FREQEST that

Theorem G.4 (High-accuracy algorithm for one-dimensional Signal Estimation) . For η ∈ R , let Λ( B ) ⊂ R denote the lattice Λ( B ) = { cη | c ∈ Z } . Suppose that f 1 , f 2 , · · · , f k ∈ Λ( B ) . Let x ∗ ( t ) = ∑ k j =1 v j exp(2 π i f j t ) , and let g ( t ) denote the noise. Given observations of the form x ( t ) = x ∗ ( t ) + g ( t ) , t ∈ [0 , T ] . Let η = min i = j | f j -f i | .

- takes S freq samples,
- runs in T freq -time, and
- outputs a set L of frequencies such that, for each f i , there exists an f ′ i ∈ L with | f i -f ′ i | ≤ D/T , holds with probability 0 . 99 .

Then, there is an algorithm (Algorithm 9) such that

- takes O ( ε -1 ˜ k log( ˜ k ) + S ) samples, · runs O ( ε -1 k ω log( k ) + T ) time,

<!-- formula-not-decoded -->

- ˜ ˜ · outputs y ( t ) = ∑ ˜ k j =1 v ′ j · exp(2 π i f ′ j t ) with ˜ k = O ( |L| (1 + D/ ( Tη ))) such that with probability at least 0 . 9 , we have

Remark G.5. For simplicity, we state the constant failure probability. It is straightforward to get failure probability ρ by blowing up a log(1 /ρ ) factor in both samples and running time.

Proof. Let L be the set of frequencies output by the Frequency Estimation algorithm FREQEST. We have the guarantee that with probability 0.99, for each true frequency f i , there exists an f ′ i ∈ L with | f i -f ′ i | ≤ D/T . Conditioning on this event, we define a set ˜ L as follows: ˜ L := { f ∈ Λ( B ) | ∃ f ′ ∈ L, | f ′ -f | &lt; D/T } . Since we assume that { f 1 , . . . , f k } ⊂ Λ( B ) , we have { f 1 , . . . , f k } ⊂ ˜ L . We use ˜ k to denote the size of set ˜ L , and we denote the frequencies in ˜ L by ˜ f 1 , ˜ f 2 , · · · , ˜ f ˜ k . Next, we need to recover magnitude v ′ ∈ C ˜ k .

<!-- formula-not-decoded -->

We first run Procedure WEIGHTEDSKETCH in Algorithm 7 and obtain a set S = { t 1 , t 2 , · · · , t s } ⊂ [0 , T ] of size s = O ( ε -2 ˜ k log( ˜ k )) and a weight vector w ∈ R s &gt; 0 . Then, we sample the signal at t 1 , . . . , t s and let x ( t 1 ) , . . . , x ( t s ) be the samples. Consider the following weighted linear regression problem:

where √ w := ( √ w 1 , . . . , √ w s ) , and the coefficients matrix A ∈ C s × ˜ k and the target vector b ∈ C s are defined as follows:

˜ ˜ ˜ Note that if v ′ corresponds to the true coefficients v , then we have ∥ √ w ◦ ( Av ′ -b ) ∥ 2 = ∥ √ w ◦ g ( S ) ∥ 2 = ∥ g ∥ S,w . Let v ′ be the exact solution of the weighted linear regression in Eq. (12), i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

And we define the output signal to be:

<!-- formula-not-decoded -->

The estimation error guarantee ∥ y ( t ) -x ∗ ( t ) ∥ T ≤ (1 + ε ) ∥ g ( t ) ∥ T follows from Lemma G.7. The running time follows from Lemma G.6.

The theorem is then proved.

Algorithm 9 Signal estimation algorithm for one-dimensional signals (high-accuracy version)

- 1: procedure SIGNALESTIMATIONACC( x, ε, k, F, T, B )

▷

Theorem G.4

- ˜ 4: s, { t 1 , t 2 , · · · , t s } , w ← WEIGHTEDSKETCH ( ˜ k, √ ε, T, B ) ▷ ˜ k , w ∈ R ˜ k , Algorithm 7 5: A i,j ← exp(2 π i f ′ j t i ) , A ∈ C s × ˜ k
- 2: L ← FREQEST ( x, k, D, F, T, B ) 3: { f ′ 1 , f ′ 2 , · · · , f ′ k } ← { f ∈ Λ( B ) | ∃ f ′ ∈ L, | f ′ -f | &lt; D/T }
- 6: b ← ( x ( t 1 ) , x ( t 2 ) , · · · , x ( t s )) ⊤
- 7: Solving the following weighted linear regression

<!-- formula-not-decoded -->

- 8: return y ( t ) = ∑ ˜ k j =1 v ′ j · exp(2 π i f ′ j t ) . 9: end procedure

Lemma G.6 (Running time of Algorithm 9) . Algorithm 9 takes O ( ε -1 ˜ k ω log( ˜ k )) -time, giving the output of Procedure FREQEST .

Proof. At Line 7, the regression solver takes time. The remaining part of Algorithm 9 takes at most O ( s ) -time.

<!-- formula-not-decoded -->

Lemma G.7 (Estimation error of Algorithm 9) . Let y ( t ) be the output signal of Algorithm 9. With high probability, we have

<!-- formula-not-decoded -->

Proof. Let F be the family of signals with frequencies in L :

Suppose the dimension of F is m ≤ k . Let { u 1 , u 2 , · · · , u m } be an orthonormal basis of F , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand, since u i ∈ F , we can also expand these basis vectors in the Fourier basis. Let V ∈ C m × ˜ k be an linear transformation 7 such that

<!-- formula-not-decoded -->

7 When m&lt; ˜ k , V is not unique, and we take any one of such linear transformation.

- ▷ Fact A.4

Then, we have

˜ where V + ∈ C ˜ k × m is the pseudoinverse of V ; or equivalently, the i -th row of V + contains the coefficients of expanding exp(2 π i ˜ f i t ) under { u 1 , . . . , u m } . Define a linear operator α : F → C m such that for any h ( t ) = ∑ ˜ k j =1 v j exp(2 π i f j t ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define an s -bym matrix B as follows:

which gives the coefficients of h under the basis { u 1 , · · · , u ˜ k } .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

B = AV . It is easy to see that Im( B ) = Im( A ) . Thus, solving Eq. (12) is equivalent to solving:

Since y ( t ) is an solution of Eq. (12), we also know that α ( y ) is an solution of Eq. (13).

<!-- formula-not-decoded -->

For convenience, we define some notations. Let √ W := diag( √ w ) and define

<!-- formula-not-decoded -->

By Fact A.4, we know that the solution of the weighted linear regression Eq. (13) has the following closed form:

<!-- formula-not-decoded -->

Then, consider the noise in the signal. Since g is an arbitrary noise, let g ∥ be the projection of g ( x ) to F and g ⊥ = g -g ∥ be the orthogonal part to F such that

Similarly, we also define

<!-- formula-not-decoded -->

By Claim G.8, the error can be decomposed into two terms:

By Claim G.10, we have

<!-- formula-not-decoded -->

And by Claim G.13, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining them together (and re-scaling ε be an constant factor), we have that

<!-- formula-not-decoded -->

Since ∥ g ∥ ∥ 2 T + ∥ g ⊥ ∥ 2 T = ∥ g ∥ 2 T , by Cauchy-Schwarz inequality, we have that

That is,

<!-- formula-not-decoded -->

## Claim G.8 (Error decomposition) .

<!-- formula-not-decoded -->

Proof. Since y, x ∗ ∈ F and { u 1 , . . . , u ˜ k } is an orthonormal basis, we have ∥ y -x ∗ ∥ T = ∥ α ( y ) -α ( x ∗ ) ∥ 2 . Furthermore, by Eq. (14), we have α ( y ) = ( B ∗ w B w ) -1 B ∗ w · X w . And by Fact G.9, since x ∗ ∈ F , we have α ( x ∗ ) = ( B ∗ w B w ) -1 B ∗ w · X ∗ w .

Thus, we have

<!-- formula-not-decoded -->

where the second step follows from the definition of g w , the forth step follows from g w = g ∥ + g ⊥ , and the last step follows from triangle inequality.

<!-- formula-not-decoded -->

Fact G.9. For any h ∈ F , where h w = √ W [ h ( t 1 ) · · · h ( t s )] ⊤ .

<!-- formula-not-decoded -->

Proof. Suppose h ( t ) = ∑ ˜ k j =1 v j exp(2 π i ˜ f j t ) . We have

<!-- formula-not-decoded -->

where the second step follows from V + is a change of coordinates.

Hence, by the Moore-Penrose inverse, we have

<!-- formula-not-decoded -->

Claim G.10 (Bound the first term) . The following holds with high probability:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By Lemma D.2, with high probability, we have

<!-- formula-not-decoded -->

where ( S, w ) is the output of Procedure WEIGHTEDSKETCH. Conditioned on this event, by Lemma B.11,

<!-- formula-not-decoded -->

since B w is the same as the matrix A in the lemma. Hence,

<!-- formula-not-decoded -->

where the second step follows from λ max (( B ∗ w B w ) -1 ) ≤ (1 -ε ) -1 , and the third step follows from Lemma F.4 and Corollary G.12.

Lemma G.11 (Lemma 6.2 of Chen and Price (2019a)) . There exists a universal constant C 1 such that given any distribution D ′ with the same support of D and any ε &gt; 0 , the random sampling procedure with m = C 1 ( K D ′ log d + ε -1 K D ′ ) i.i.d. random samples from D ′ and coefficients α 1 = · · · = α m = 1 /m is an ε -well-balanced sampling procedure .

Corollary G.12. Procedure WEIGHTEDSKETCH in Algorithm 7 is a ε -WBSP (Definition E.1). Claim G.13 (Bound the second term) .

Proof.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step follows from Fact G.9 and g ∥ ∈ F , the second step follows from the definition of α .

## H High-Accuracy Fourier Interpolation Algorithm

In this section, we propose an algorithm for one-dimensional continuous Fourier interpolation problem, which significantly improves the accuracy of the algorithm in Chen et al. (2016).

This section is organized as follows. In Sections H.1 and H.2, we provide some technical tools for Fourier-sparse signals, low-degree polynomials and filter functions. In Section H.3, we design a high sensitivity frequency estimation method using these tools. In Section H.4, we combine the frequency estimation with our Fourier set query framework, and give a (9+ ε ) -approximate Fourier interpolation algorithm. Then, in Section H.5, we build a sharper error control, and in Section H.6, we analysis the HASHTOBINS procedure. Based on these result, in Section H.8, we develop the ultra-high sensitivity frequency estimation method. In Section H.10, we show the a (3 + √ 2 + ε ) -approximate Fourier interpolation algorithm.

## H.1 Technical tools I: Fourier-polynomial equivalence

In this section, we show that low-degree polynomials and Fourier-sparse signals can be transformed to each other with arbitrarily small errors.

The following lemma upper-bounds the error of using low-degree polynomial to approximate Fouriersparse signal.

Lemma H.1 (Fourier signal to polynomial, Chen et al. (2016)) . For any ∆ &gt; 0 and any δ &gt; 0 , let x ∗ ( t ) = ∑ j ∈ [ k ] v j e 2 π i f j t where | f j | ≤ ∆ for each j ∈ [ k ] . There exists a polynomial P ( t ) of degree at most such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As a corollary, we can expand a Fourier-sparse signal under the mixed Fourier-monomial basis (i.e., { e 2 π i f i t · t j } i ∈ [ k ] ,j ∈ [ d ] ).

<!-- formula-not-decoded -->

Corollary H.2 (Mixed Fourier-polynomial approximation) . For any ∆ &gt; 0 , δ &gt; 0 , n j ∈ Z ≥ 0 , j ∈ [ k ] , ∑ j ∈ [ k ] n j = k . Let where | f ′ j,i | ≤ ∆ for each j ∈ [ k ] , i ∈ [ n j ] . There exist k polynomials P j ( t ) for j ∈ [ k ] of degree at most

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

such that

The following lemma bounds the error of approximating a low-degree polynomial using Fouriersparse signal.

Lemma H.3 (Polynomial to Fourier signal, Chen et al. (2016)) . For any degreed polynomial Q ( t ) = d ∑ j =0 c j t j , any T &gt; 0 and any ε &gt; 0 , there always exist γ &gt; 0 and

<!-- formula-not-decoded -->

with some coefficients α 0 , · · · , α d such that

<!-- formula-not-decoded -->

## H.2 Technical tools II: filter functions

In this section, we introduce the filter functions H and G designed by Chen et al. (2016), and we generalize their constructions to achieve higher sensitivity.

We first construct the H -filter, which uses rect and sinc functions.

Fact H.4 ( rect function Fourier transform) . For s &gt; 0 , let rect s ( t ) := 1 | t |≤ s/ 2 . Then, we have

<!-- formula-not-decoded -->

Definition H.5. Given s 1 , s 2 &gt; 0 and an even number ℓ ∈ N + , we define the filter function H 1 ( t ) and its Fourier transform H 1 ( f ) as follows:

<!-- formula-not-decoded -->

where s 0 = C 0 s 1 √ ℓ is a normalization parameter such that H 1 (0) = 1 , and ⋆ means convolution.

<!-- formula-not-decoded -->

Definition H.6 ( H -filter's construction, Chen et al. (2016)) . Given any 0 &lt; s 1 , s 3 &lt; 1 , 0 &lt; δ &lt; 1 , we define H s 1 ,s 3 ,δ ( t ) from the filter function H 1 ( t ) (Definition H.5) as follows:

- let ℓ := Θ( k log( k/δ )) , s 2 := 1 -2 s 1 , and
- shrink H 1 by a factor s 3 in time domain, i.e.,

<!-- formula-not-decoded -->

̂ H s 1 ,s 3 ,δ ( f ) = s 3 ̂ H 1 ( s 3 f ) (16) We call the 'filtered cluster" around a frequency f 0 to be the support of ( δ f 0 ⋆ ̂ H s 1 ,s 3 ,δ )( f ) in the frequency domain and use

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

to denote the width of the cluster.

Lemma H.7 (High sensitivity H -filter's properties) . Given ε ∈ (0 , 0 . 1) , s 1 , s 3 ∈ (0 , 1) with min( 1 1 -s 3 , s 1 ) ≥ ˜ O ( k 4 ) /ε , and δ ∈ (0 , 1) . Let the filter function H := H s 1 ,s 3 ,δ ( t ) defined in Definition H.6. Then, H satisfies the following properties:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any exact k -Fourier-sparse signal x ∗ ( t ) , we shift the interval from [0 , T ] to [ -1 / 2 , 1 / 2] and consider x ∗ ( t ) for t ∈ [ -1 / 2 , 1 / 2] to be our observation, which is also x ∗ ( t ) · rect 1 ( t ) .

Remark H.8. By Property I, and II, and III, we have that H ( t ) ≤ 1 for t ∈ [0 , T ] .

Proof. The proof of Property I - V easily follows from Chen et al. (2016). We prove Property VI in below.

First, because of for any t , | H 1 ( t ) | ≤ 1 , thus we prove the upper bound for LHS,

<!-- formula-not-decoded -->

Second, as mentioned early, we need to prove the general case when s 3 = 1 -1 / poly( k ) . Define interval S = [ -s 3 ( 1 2 -1 s 1 ) , s 3 ( 1 2 -1 s 1 )] , by definition, S ⊂ [ -1 / 2 , 1 / 2] . Then define S = [ -1 / 2 , 1 / 2] \ S , which is [ -1 / 2 , -s 3 ( 1 2 -1 s 1 )) ∪ ( s 3 ( 1 2 -1 s 1 ) , 1 / 2] . By Property I, we have

Then we can show

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step follows from S ⊂ [ -1 / 2 , 1 / 2] , the second step follows from Theorem C.1, the third step follows from (1 -s 3 (1 -2 s 1 )) · O ( k 2 ) ≤ ε .

Combining Equations (18) and (19) gives a lower bound for LHS,

<!-- formula-not-decoded -->

where the first step follows from S ⊂ [ -1 / 2 , 1 / 2] , the second step follows from Eq. (18), the third step follows from S ∩ S = ∅ , the forth step follows from Eq. (19), the fifth step follows from S ∪ S = [ -1 / 2 , 1 / 2] , the last step follows from ε ≫ δ .

As remarked in Chen et al. (2016), to match ( H ( t ) , ̂ H ( f )) on [ -1 / 2 , 1 / 2] with signal x ( t ) on [0 , T ] , we will scale the time domain from [ -1 / 2 , 1 / 2] to [ -T/ 2 , T/ 2] and shift it to [0 , T ] . Then, in frequency domain, the Property IV in Lemma H.7 becomes

<!-- formula-not-decoded -->

We also need another filter function, G , whose construction and properties are given below.

Definition H.9 ( G -filter's construction, Chen et al. (2016)) . Given B &gt; 1 , δ &gt; 0 , α &gt; 0 . Let l := Θ(log( k/δ )) . Define G B,δ,α ( t ) and its Fourier transform ̂ G B,δ,α ( f ) as follows:

<!-- formula-not-decoded -->

where b 0 = Θ( B √ l/α ) is the normalization factor such that G (0) = 1 .

<!-- formula-not-decoded -->

̂ Lemma H.10 ( G -filter's properties, Chen et al. (2016)) . Given B &gt; 1 , δ &gt; 0 , α &gt; 0 , let G := G B,δ,α ( t ) be defined in Definition H.9. Then, G satisfies the following properties:

$$Property I : ̂ G ( f ) ∈ [1 - δ/k, 1] , if | f | ≤ (1 - α ) 2 B . Property II : ̂ G ( f ) ∈ [0 , 1] , if (1 - α ) 2 π 2 B ≤ | f | ≤ 2 π 2 B . Property III : ̂ G ( f ) ∈ [ - δ/k, δ/k ] , if | f | > 2 π 2 B . Property IV : supp( G ( t )) ⊂ [ l 2 · - B πα , l 2 · B πα ] . Property V : max G ( t ) ≲ poly( B,l ) .$$

$$2 π$$

t | |

## H.3 High sensitivity frequency estimation

In this section, we show a high sensitivity frequency estimation. Compared with the result in Chen et al. (2016), we relax the condition of the frequencies that can be recovered by the algorithm.

Definition H.11 (Definition 2.4 in Chen et al. (2016)) . Given x ∗ ( t ) = k ∑ j =1 v j e 2 π i f j t , any N &gt; 0 , and a filter function H with bounded support in frequency domain. Let L j denote the interval of supp( ̂ e 2 π i f j t · H ) for each j ∈ [ k ] . Define an equivalence relation ∼ on the frequencies f i as follows:

<!-- formula-not-decoded -->

̸

Let S 1 , . . . , S n be the equivalence classes under this relation for some n ≤ k .

<!-- formula-not-decoded -->

Define C i := ∪ f ∈ S i L i for each i ∈ [ n ] . We say C i is an N -heavy cluster iff

The following claim gives a tight error bound for approximating the true signal x ∗ ( t ) by the signal x S ∗ ( t ) whose frequencies are in heavy-clusters. It improves the Claim 2.5 in Chen et al. (2016).

Claim H.12 (Approximation by heavy-clusters) . Given x ∗ ( t ) = k ∑ j =1 v j e 2 π i f j t and any N &gt; 0 , let C 1 , · · · , C l be the N -heavy clusters from Definition H.11. For

∣ we have x S ∗ ( t ) = ∑ j ∈ S ∗ v j e 2 π i f j t approximating x ∗ within distance

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let H be the filter function defined as in Definition H.6.

Let

<!-- formula-not-decoded -->

Notice that ∥ x ∗ -x S ∗ ∥ 2 T = ∥ x S ∗ ∥ 2 T .

By Property VI in Lemma H.7 with setting ε = ε 0 := ε/ 2 , we have

<!-- formula-not-decoded -->

where the first step follows from the definition of the norm, the second step follows from the definition of rect T ( t ) = 1 , ∀ t ∈ [0 , T ] , the third step follows from Lemma H.7, the forth step follows from rect T ( t ) ≤ 1 .

From Definition H.11, we have

<!-- formula-not-decoded -->

where the first step follows from Parseval's theorem, the second step follows from Definition H.11, Property IV of Lemma H.7, the definition of S ∗ , thus, supp( ̂ x S ∗ · H ( f )) = C 1 ∪ · · · ∪ C l , supp( ̂ x S ∗ · H ( f )) ∩ supp( ̂ x S ∗ · H ( f ))) = ∅ , the last step follows from Definition H.11.

Overall, we have (1 -ε 0 ) ∥ x S ∗ ∥ 2 T ≤ N 2 . Thus, ∥ x S ∗ ( t ) -x ∗ ( t ) ∥ 2 T ≤ (1 -l/k )((1 + ε ) N 2 by the basic algebra fact: 1 1 -ε/ 2 ≤ 1 + ε for any ε ∈ [0 , 1] .

Due to the noisy observations, not all frequencies in heavy-clusters are recoverable. Thus, we define the recoverable frequency as follows:

Definition H.13 (Recoverable frequency) . Let C be an N 1 -heavy cluster. We say C is N 2 -recoverable if it satisfies:

<!-- formula-not-decoded -->

A frequency f is ( N 1 , N 2 ) -recoverable if f is in an N 1 -heavy, N 2 -recoverable cluster C .

The following lemma shows that most heavy clusters are also recoverable.

Lemma H.14 (Heavy-clusters are almost recoverable) . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t and x ( t ) = x ∗ ( t ) + g ( t ) be our observable signal. Let N 2 := ∥ g ∥ 2 T + δ ∥ x ∗ ∥ 2 T . Let C 1 , · · · , C l are the 2 N -heavy clusters from Definition H.11. Let S ∗ denotes the set of frequencies f ∗ ∈ { f j } j ∈ [ k ] such that, f ∗ ∈ C i for some i ∈ [ l ] . Let S ⊂ S ∗ be the set of (2 N , N ) -recoverable frequencies.

Then we have that,

<!-- formula-not-decoded -->

Proof. If a cluster C i is 2 N -heavy but not N -recoverable, then it holds that:

where the first steps follows from C i ⊂ ⋃ f j ∈ S ∗ C j , the second step follows from C i ̸⊂ ⋃ f j ∈ S C j . So,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step follows from g ( t ) = x ( t ) -x ∗ ( t ) , and the second step follows from triangle inequality, the last step follows from Eq. (21).

<!-- formula-not-decoded -->

Let C ′ := ⋃ f j ∈ S ∗ \ S C j , i.e., the union of heavy but not recoverable clusters. Then, we have

̸

where the first step follows from the definition of the norm and C i ∩ C j = ∅ , ∀ i = j , the second step follows from Eq. (22).

Then we have that

<!-- formula-not-decoded -->

where the first step follows from Property VI of H in Lemma H.7 (taking ε there to be ε/ 2 ), the second step follows from ε ∈ [0 , 1] and the definition of C i , the third step follows from Eq. (23), the forth step follows from g ( t ) = 0 , ∀ t ̸∈ [0 , T ] , the fifth step follows from Remark H.8, the last step follows from the definition of N 2 . Thus, we get that:

<!-- formula-not-decoded -->

which follows from √ 1 + ε ≤ 1 + ε/ 2 .

Finally, we can conclude that

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows from the definition of x S ∗\ S , the third step follows from Claim H.12, the last step follows from Eq. (24). The lemma follows by re-scaling ε to ε/ 2 .

## H.4 (9 + ε ) -approximate Fourier interpolation algorithm

The goal of this section is to prove Theorem H.20, which gives a Fourier interpolation algorithm with approximation error (9 + ε ) . It improves the constant (more than 1000) error algorithm in Chen et al. (2016).

Claim H.15 (Mixed Fourier-polynomial energy bound, Chen et al. (2016)) . For any we have that

<!-- formula-not-decoded -->

Claim H.16 (Condition number of Mixed Fourier-polynomial) . Let F is a linear function family as follows:

<!-- formula-not-decoded -->

Then the condition number of Uniform[0 , T ] with respect to F is as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following definition extends the well-balanced sampling procedure (Definition E.1) to high probability.

Definition H.17 (( ε, ρ )-well-balanced sampling procedure) . Given a linear family F and underlying distribution D , let P be a random sampling procedure that terminates in m iterations ( m is not necessarily fixed) and provides a coefficient α i and a distribution D i to sample x i ∼ D i in every iteration i ∈ [ m ] .

We say P is an ε -WBSP if it satisfies the following two properties:

<!-- formula-not-decoded -->

2. The coefficients always have ∑ m i =1 α i ≤ 5 4 and α i · K IS ,D i ≤ ε 2 for all i ∈ [ m ] . The following lemma shows an ( ε, ρ ) -WBSP for mixed Fourier-polynomial family.

Lemma H.18 (WBSP for mixed Fourier-polynomial family) . Given any distribution D ′ with the same support of D and any ε &gt; 0 , the random sampling procedure with m = O ( ε -1 K IS ,D ′ log( d/ρ )) i.i.d. random samples from D ′ and coefficients α 1 = · · · = α m = 1 /m is an ( ε, ρ ) -WBSP.

Proof. By Lemma B.12 with setting ε = √ ε , we have that, as long as m ≥ O ( 1 ε · K IS ,D ′ log d ρ ) , then with probability 1 -ρ ,

<!-- formula-not-decoded -->

By Lemma B.11, we have that, for every h ∈ F , where S is the m i.i.d. random samples from D ′ , w i = α i D ( x i ) /D ′ ( x i ) .

<!-- formula-not-decoded -->

Moreover, ∑ m i =1 α i = 1 ≤ 5 / 4 and

<!-- formula-not-decoded -->

where the first step follows from the definition of α i , the second step follows from the definition of m , the third step follows from log( d/ρ ) &gt; 1 .

Now, we can solve the Signal Estimation problem for mixed Fourier-polynomial signals.

Lemma H.19 (Mixed Fourier-polynomial signal estimation) . Given d -degree polynomials P j ( t ) , j ∈ [ k ] and frequencies f j , j ∈ [ k ] . Let x S ( t ) = ∑ k j =1 P j ( t ) exp(2 π i f j t ) , and let g ( t ) denote the noise. Given observations of the form x ( t ) := x S ( t )+ g ′ ( t ) for arbitrary noise g ′ in time duration t ∈ [0 , T ] .

Then, there is an algorithm such that

- takes O ( ε -1 poly( kd ) log(1 /ρ )) samples from x ( t ) ,
- runs O ( ε -1 poly( kd ) log(1 /ρ )) time,
- outputs y ( t ) = ∑ k j =1 P ′ j ( t ) exp(2 π i f j t ) with d -degree polynomial P ′ j ( t ) , such that with probability at least 1 -ρ , we have

<!-- formula-not-decoded -->

Proof sketch. The proof is almost the same as Theorem G.4 where we follow the four-step Fourier set-query framework. Claim H.15 gives the energy bound for the family of mixed Fourier-polynomial signals, which implies that uniformly sampling m = ˜ O ( ε -1 | L | 4 d 4 ) points in [0 , T ] forms an oblivious sketch for x ∗ . Moreover, by Lemma H.18, we know that it is also an ( ε, ρ ) -WBSP, which gives the error guarantee. Then, we can obtain a mixed Fourier-polynomial signal y ( t ) by solving a weighted linear regression.

Now, we are ready to prove the main result of this section, a (9+ ε ) -approximate Fourier interpolation algorithm.

Theorem H.20 (Fourier interpolation with (9 + ε ) -approximation error) . Let x ( t ) = x ∗ ( t ) + g ( t ) , where x ∗ is k -Fourier-sparse signal with frequencies in [ -F, F ] . Given samples of x over [0 , T ] we can output y ( t ) such that with probability at least 1 -2 -Ω( k ) ,

Our algorithm uses poly( k, ε -1 , log(1 /δ )) log( FT ) samples and poly( k, ε -1 , log(1 /δ )) · log 2 ( FT ) time. The output y is poly( k, log(1 /δ )) ε -1 . 5 -Fourier-sparse signal.

<!-- formula-not-decoded -->

Proof. Let N 2 := ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T be the heavy cluster parameter.

<!-- formula-not-decoded -->

First, by Lemma H.14, there is a set of frequencies S ⊂ [ k ] and x S ( t ) = ∑ j ∈ S v j e 2 π i f j t such that

Furthermore, each f j with j ∈ S belongs to an N -heavy cluster C j with respect to the filter function H defined in Definition H.6.

By Definition H.11 of heavy cluster, it holds that

<!-- formula-not-decoded -->

By Definition H.11, we also have | C j | ≤ k · ∆ h , where ∆ h is the bandwidth of ̂ H . Let ∆ ∈ R + , and ∆ &gt; k · ∆ h , which implies that C j ⊆ [ f j -∆ , f j +∆] . Thus, we have

<!-- formula-not-decoded -->

Now it is enough to recover only x S , instead of x ∗ .

By applying Theorem H.36, there is an algorithm that outputs a set of frequencies L ⊂ R such that, | L | = O ( k ) , and with probability at least 1 -2 -Ω( k ) , for any f j with j ∈ S f , there is a ˜ f ∈ L such that,

<!-- formula-not-decoded -->

We define a map p : R → L as follows:

<!-- formula-not-decoded -->

Then, x S ( t ) can be expressed as

<!-- formula-not-decoded -->

where the first step follows from the definition of x S , the last step follows from interchanging the summations.

<!-- formula-not-decoded -->

For each ˜ f i ∈ L , by Corollary H.2 with x ∗ = x S f , ∆ = ∆ √ ∆ T , we have that there exist degree d = O ( T ∆ √ ∆ T + k 3 log k + k log 1 /δ ) polynomials P i ( t ) corresponding to f i ∈ L such that,

Define the following function family:

<!-- formula-not-decoded -->

By Claim H.16, for function family F , K Uniform[0 , T] = O (( | L | d ) 4 log 3 ( | L | d )) .

Note that ∑ ˜ f i ∈ L e 2 π i ˜ f i t P i ( t ) ∈ F .

By Lemma H.18, we have that, choosing a set W of O ( ε -1 K Uniform[0 , T] log( | L | d/ρ )) i.i.d. samples uniformly at random over duration [0 , T ] is a ( ε, ρ ) -WBSP.

By Lemma H.19, there is an algorithm that runs in O ( ε -1 | W | ( | L | d ) ω -1 log(1 /ρ )) -time using samples in W , and outputs y ′ ( t ) ∈ F such that, with probability 1 -ρ ,

<!-- formula-not-decoded -->

Then by Lemma H.3, we have that there is a O ( kd ) -Fourier-sparse signal y ( t ) , such that

<!-- formula-not-decoded -->

Moreover, the sparsity of y ( t ) is kd = kO ( T ∆ √ ∆ T + k 3 log k + k log 1 /δ ) = ε -1 . 5 poly( k, log(1 /δ )) .

where δ ′ &gt; 0 is any positive real number, thus, y can be arbitrarily close to y ′ .

Therefore, the total approximation error can be upper bounded as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

≤ (1 + 2 ε ) ∥ g ∥ T +(2 + ε ) ˜ f i ∈ L e 2 π i ˜ f i t P i ( t ) -x ∗ ∥ T (Triangle inequality) ≤ (1 + 2 ε ) ∥ g ∥ T +(2 + ε ) ∥ ∥ ∥ ∑ ˜ f i ∈ L e 2 π i ˜ f i t P i ( t ) -x S f ∥ ∥ ∥ T +(2 + ε ) ∥ x S f -x ∗ ∥ T (Triangle inequality) ≤ (1 + 2 ε ) ∥ g ∥ T +(2 + ε ) δ ∥ x S f ∥ T +(2 + ε ) ∥ x S f -x ∗ ∥ T (Eq. (26)) ≤ (1 + 2 ε ) ∥ g ∥ T + O ( δ ) ∥ x ∗ ∥ T +(2 + ε )(1 + δ ) ∥ x S f -x ∗ ∥ T (Triangle inequality) ≤ (1 + 2 ε ) ∥ g ∥ T + O ( δ ) ∥ x ∗ ∥ T +(2 + ε )(1 + δ )( ∥ x S f -x S ∥ T + ∥ x S -x ∗ ∥ T ) (Triangle inequality) ≤ (1 + 2 ε ) ∥ g ∥ T + O ( δ ) ∥ x ∗ ∥ T +(2 + ε + O ( δ ))(4 + O ( ε )) N (Eq. (25) and Lemma H.41) =(1 + 2 ε ) ∥ g ∥ T + O ( δ ) ∥ x ∗ ∥ T +(8 + O ( ε + δ )) N ,

Since we take we have

<!-- formula-not-decoded -->

By re-scaling ε and δ , we prove the theorem.

<!-- formula-not-decoded -->

## H.5 Sharper error control by signal-noise cancellation effect

In this section, we significantly improve the error analysis in Section H.3. Our key observation is the signal-noise cancellation effect : if there is a frequency f ∗ in a N 1 -heavy cluster but not ( N 1 , N 2 ) -recoverable for some N 2 &lt; N 1 , then it indicates that the contribution of f ∗ to the signal x ∗ 's energy are cancelled out by the noise g .

In the following lemma, we improving Lemma H.14 by considering g 's effect in the gap between heavy-cluster signal and recoverable signal.

Lemma H.21 (Sharper error bound for recoverable signal, an improved version of Lemma H.14) . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t and x ( t ) = x ∗ ( t ) + g ( t ) be our observable signal. Let N 2 1 := ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T . Let C 1 , · · · , C l are the N 1 -heavy clusters from Definition H.11. Let S ∗ denotes the set of frequencies f ∗ ∈ { f j } j ∈ [ k ] such that, f ∗ ∈ C i for some i ∈ [ l ] . Let S ⊂ S ∗ be the set of ( N 1 , √ ε 2 N 1 ) -recoverable frequencies (Definition H.13).

Then we have that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In order for cluster C i to be missed, we must have that where the first steps follows from C i ⊂ ∪ f j ∈ S ∗ C j , the second step follows from C i ̸⊂ ∪ f j ∈ S C j . Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step follows from the definition of g ′ , the second step follows from triangle inequality, the third step follows from Eq. (29), the last step follows from ε 2 ≤ 0 . 1 .

Bound ∥ H · x -H · x S ∥ T . Let I ′ = ∪ f j ∈ S ∗ \ S C j , then we have that, where the first step follows from the definition of the norm, the second step follows from Parseval's theorem, the third step follows from I ′ ∪ I ′ = [ -∞ , ∞ ] .

<!-- formula-not-decoded -->

Bound ∥ H · x S ∗ -H · x S ∥ T We can upper-bound it as follows:

where the first step follows from the definition of the norm, the second step follows from Parseval's theorem, the third step follows from I ′ ∪ I ′ = [ -∞ , ∞ ] , the last step follows from ( ∪ f j ∈ S ∗ /S C j ) ∩ I ′ = ∅ .

<!-- formula-not-decoded -->

Putting it all together. By Eqs. (31) and (32), we get that

<!-- formula-not-decoded -->

For the first integral, we have where the first step follows from ( ∪ f j ∈ S C j ) ∩ I ′ = ∅ , the second step follows from triangle inequality, the third step follows from Eq. (30), the last step is straightforward.

<!-- formula-not-decoded -->

For the second integral, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step follows from ( ∪ f j ∈ S C j ) ∩ I ′ = ∅ , the second step follows from Eq. (30).

<!-- formula-not-decoded -->

For the third integral, together with the ∫ I ′ | ̂ H · g ′ ( f ) | 2 d f term in the first integral's upper bound (Eq. (33)), we have where the first step follows from the definition of g ′ , the second step follows from ( ∪ f j ∈ S ∗ C j ) ∩ I ′ = ( ∪ f j ∈ S C j ) , the third step follows from I ′ ∪ I ′ = [ -∞ , ∞ ] , the forth step follows from Parseval's theorem, the fifth step follows from g ′ ( t ) = 0 , ∀ t ̸∈ [0 , T ] , the last step follows from H ( t ) ≤ 1 by Remark H.8.

Furthermore, we have that

<!-- formula-not-decoded -->

Therefore, we conclude that

<!-- formula-not-decoded -->

where the first step follows from Eq. (31), the second step follows from Eq. (32), the third step follows from Eq. (33), the forth step follows from Eq. (34), the fifth step follows from (1+ √ 2 ε 2 ) 2 ≤ 1 + O ( √ ε 2 ) , the sixth step follows from Eq. (36), the seventh step follows from Eq. (35), the last step is straightforward.

The lemma is then proved.

As a consequence, we can easily bound ∥ x S ∗ -x S ∥ T as follows.

Corollary H.22. Let S ∗ and S be defined as in Lemma H.21. Then, we have that,

<!-- formula-not-decoded -->

Proof. We have that,

<!-- formula-not-decoded -->

where the first step follows from Lemma H.7 Property VI, the second step follows from Lemma H.21 and ε = ε 2 .

In Lemma H.21, we introduce an extra term ∥ H · x -H · x S ∥ T . The following lemma shows that this term appears in the approximation error ∥ x -x S ∥ T , which can be used to upper-bound the Signal Estimation's error.

Lemma H.23 (Decomposing the approximation error of recoverable signal) . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t and x ( t ) = x ∗ ( t ) + g ( t ) be our observable signal. Let N 2 1 := ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T . Let C 1 , · · · , C l are the N 1 -heavy clusters from Definition H.11. Let S ∗ denotes the set of frequencies f ∗ ∈ { f j } j ∈ [ k ] such that, f ∗ ∈ C i for some i ∈ [ l ] , and

<!-- formula-not-decoded -->

Let S denotes the set of frequencies f ∗ ∈ S ∗ such that, f ∗ ∈ C j for some j ∈ [ l ] , and

Then we have that,

<!-- formula-not-decoded -->

Proof. We first decompose ∥ x -x S ∥ T into the part that passes through the filter H and the part that does not pass through H :

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows from triangle inequality, the last step follows from the definition of g .

For the second term, we have that by Remark H.8.

For the third term, we have that,

<!-- formula-not-decoded -->

where the first step follows from 1 -H &gt; 0 , the second step follows from x ∗ -x S is k -Fourier-sparse, thus combine Property VI of Lemma H.7, we have that ∥ H ( x ∗ -x S ) ∥ 2 T ≥ (1 -ε ) ∥ x ∗ -x S ∥ 2 T .

Combining them together, we prove the lemma.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## H.6 Technical tools III: HASHTOBINS

In this section, we provide some definitions and technical lemmas for the HASHTOBINS procedure, which will be very helpful for frequency estimation.

HASHTOBINS partitions the frequency coordinates into B = O ( k ) bins and collects rotated magnitudes in each bins. Ideally, each bins only contains a single ground-truth frequency, which allows us to recover its magnitude.

More specifically, HASHTOBINS first randomly hashes the frequency coordinates into the interval [0 , 1] . After equally dividing [0 , 1] into O ( k ) small bins, each coordinate lays in a different bin. This step can be implemented by multiplying the signal in the frequency domain with a period pulse function G ( j ) σ,b . Then, even if the signal does not have frequency gap, the HASHTOBINS procedure can still partition it into several one-cluster signals with high probability.

Definition H.24 (Hash function, Chen et al. (2016)) . Let π σ,b ( f ) = σ ( f + b ) (mod 1) and h σ,b ( f ) = round( π σ,b ( f ) · B ) be the hash function that maps frequency f ∈ [ -F, F ] into bins { 0 , · · · , B -1 } .

Claim H.25 (Collision probability, Chen et al. (2016)) . For any ∆ 0 &gt; 0 , let σ be a sample uniformly at random from [ 1 4 B ∆ 0 , 1 2 B ∆ 0 ] . Then, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Definition H.26 (Filter for bins) . Given B &gt; 1 , δ &gt; 0 , α &gt; 0 , let G ( t ) := G B,δ,α (2 πt ) where G B,δ,α is defined in Definition H.9. For any σ &gt; 0 , b ∈ R and j ∈ [ B ] . define and its Fourier transformation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Definition H.27 ( ( ε 0 , ∆ 0 ) -one-cluster signal, Chen et al. (2016)) . We say that a signal z ( t ) is an ( ε 0 , ∆ 0 ) -one-cluster signal around f 0 iff z ( t ) and z ( f ) satisfy the following two properties:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Definition H.28 (Well-isolation, Chen et al. (2016)) . We say that a frequency f ∗ is well-isolated under the hashing ( σ, b ) if, for j = h σ,b ( f ∗ ) and I f ∗ = ( -∞ , ∞ ) \ ( f ∗ -∆ 0 , f ∗ +∆ 0 ) , where N 2 2 := ε 1 ε 2 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) .

<!-- formula-not-decoded -->

Lemma H.29 (Well-isolation implies one-cluster signal, a variation of Lemma 7.20 in Chen et al. (2016)) . Let f ∗ satisfy where N 2 2 := ε 1 ε 2 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) . Let ̂ z = ̂ x ∗ · H · ̂ G ( j ) σ,b where j = h σ,b ( f ∗ ) . If f ∗ is well-isolated, then z and ̂ z satisfying Property II of one-cluster signal (Definition H.27), i.e.,

LemmaH.30 (Well-isolation by randomized hashing, Chen et al. (2016)) . Given B = Θ( k/ ( ε 0 ε 1 ε 2 )) and σ ∈ [ 1 4 B ∆ 0 , 1 2 B ∆ 0 ] chosen uniformly at random. Let f ∗ be any frequency. Then f ∗ is wellisolated by a hashing ( σ, b ) with probability at least 0 . 9 .

<!-- formula-not-decoded -->

̸

Proof. Let S ′ = { f i } i ∈ [ k ] ∩ I f ∗ . By Claim H.25, with probability at least (1 -1 /B ) k ≥ 1 -k/B ≥ 1 -ε 0 ε 1 ε 2 ≥ 0 . 99 , for all the frequencies f ∈ S ′ , we have that h σ,b ( f ∗ ) = h σ,b ( f ) . Hence,

<!-- formula-not-decoded -->

where the first step follows by the Property III in the Lemma H.10 that | ̂ G ( f ) | ≤ δ/k , which implies that | ̂ G ( j ) σ,b ( f ) | ≤ O ( δ/k ) for f ∈ S ′ , the second step follows from I f ∗ ⊂ [ -∞ , ∞ ] , the third step follows from Parseval's theorem, the forth step is straight forward, the fifth step follows from the property VI of Lemma H.7, the sixth step follows from V of Lemma H.7.

Moreover, let I ′ denote the set of frequencies that hash into the same bin as f ∗ , then we have that,

<!-- formula-not-decoded -->

where the first step follows from I ′ ∪ I ′ = [ -∞ , ∞ ] , the second step follows from for any f ∈ R , ̂ G ( j ) σ,b ( f ) ≲ 1 , the third step follows from for any f ∈ I ′ , G ( j ) σ,b ( f ) ≲ δ/k , the last step follows from

<!-- formula-not-decoded -->

where the first step follows from I ′ ∈ [ -∞ , ∞ ] , the second step follows from Parseval's theorem, the third step follows from g ( t ) = 0 , ∀ t ̸∈ [0 , T ] , the last step follows from Remark H.8.

Next, we consider

<!-- formula-not-decoded -->

where the first step follows from σ, b are chosen randomly, the second step follows from ∫ ∞ -∞ | ̂ g · H ( f ) | 2 d f ≤ T ∥ g ∥ 2 T . Thus, by Markov inequality, with probability at least 0 . 99 ,

<!-- formula-not-decoded -->

Finally, we can conclude that

<!-- formula-not-decoded -->

where the first step follows from the definition of g , the second step follows from ( a + b ) 2 ≤ 2 a 2 +2 b 2 , the third step follows from Eq. (37), the forth step follows from Eq. (38), the fifth step follows from Eq. (39), the sixth step is straightforward, the seventh step follows from δ (1+ δ ) ε 0 ε 1 ε 2 k ≤ 1 and ( δ 2 ε 0 ε 1 ε 2 k +1) ≤ 2 , the last step follows from the definition of N 2 2 .

Lemma H.31 ((Chen et al., 2016, Lemma 7.21)) . Given any noise g ( t ) : [0 , T ] → C and g ( t ) = 0 , ∀ t / ∈ [0 , T ] . We have, ∀ j ∈ [ B ] ,

<!-- formula-not-decoded -->

## H.7 High signal-to-noise ratio (SNR) band approximation

In the this section, we will give the upper bound of ∥ x S f ( t ) -x S ( t ) ∥ T .

Definition H.32 (High SNR and Recoverable Set) . For j ∈ [ B ] , let z ∗ j ( t ) := ( x ∗ · H ) · G ( j ) σ,b , we define the set as follows

<!-- formula-not-decoded -->

where c is constant. And we also give the definition of recoverable set which is the same with s above

<!-- formula-not-decoded -->

where N 2 2 := ε 1 ε 2 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T .

And then we define a High SNR and recoverable set as follows

<!-- formula-not-decoded -->

Remark H.33. In the left part of the paper, we focus on the frequency in set S f which is a subset of the recoverable frequency set S .

<!-- formula-not-decoded -->

The following lemma shows that for any recoverable frequency (i.e., those satisfy Eq. (40)), HASHTOBINS will output a one-cluster signal around it with high probability. Now we will consider a f ∗ satisfy the assumption introduced in Definition H.32.

Lemma H.34 (HASHTOBINS for recoverable and HSR frequency) . Let f ∗ ∈ [ -F, F ] satisfy:

<!-- formula-not-decoded -->

where N 2 2 := ε 1 ε 2 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) .

Proof. The proof consists of two parts. In part 1, we prove that z ( t ) satisfies Property I of the one-cluster signal around f ∗ (Definition H.27). In part 2, we prove that z ( t ) satisfies Property II of Definition H.27.

For a random hashing ( σ, b ) , let j = h σ,b ( f ∗ ) be the bucket that f ∗ maps to under the hash such that z = ( x · H ) ∗ G ( j ) σ,b and ̂ z = ̂ x · H · ̂ G ( j ) σ,b . Given that S f and c is defined in Definition H.32, j ∈ S f . With probability at least 0 . 9 , z ( t ) is an ( ε 0 , ∆ 0 ) -one-cluster (See Definition H.27) signal around f ∗ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, with probability at least 0 . 99 , we have that where the probability follows from ∆ 0 &gt; 1000∆ , the first step follows from Property I of G in Lemma H.10, the second step follows from Eq. (40).

On the other hand, f ∗ is well-isolated with probability 0 . 9 , thus by the definition of well-isolated, we have that

Hence, ̂ z satisfies the Property I (in Definition H.27) of one-mountain recovery. Part 2. By Lemma H.29, we know that ( x ∗ · H ) ∗ G ( j ) σ,b always satisfies Property II (in Definition H.27):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, we claim that where the first step follows from Parseval's theorem, the second step follows from [ f ∗ -∆ , f ∗ +∆] ⊂ [ -∞ , ∞ ] , the third step holds with probability at least 0 . 99 and follows from ∆ 0 &gt; 1000∆ and Property I of Lemma H.10, the last step follows from the definition of f ∗ .

<!-- formula-not-decoded -->

By Definition H.32, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step from g ( t ) = 0 , ∀ t ̸∈ [0 , T ] , the second step follows from Definition H.32, the third step follows from simple algebra, the last step is due to Definition of z ∗ j ( t ) .

Then, we claim that

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows from Eq. (43). Next, we consider

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows from Eq. (41), the third step follows from Eq. (43), the forth step follows from Eq. (44). Similarly,

<!-- formula-not-decoded -->

Combine equations above, we have that, where the first step follows from √ a + b ≤ √ a + √ b , the second step follows from Eq. (45) and Eq. (46).

<!-- formula-not-decoded -->

Hence, we have that z = ( x ∗ + g ) · H ∗ G ( j ) σ,b satisfies Property II (in Definition H.27) with probability 0 . 95 .

## H.8 Ultra-high sensitivity frequency estimation

In this section, we improve the high sensitivity frequency estimation in Section H.3 with even higher sensitivity, using the results in previous sections. More specifically, we show how to estimate the frequencies of the signal x S whose frequencies are only ε 2 N -heavy, while in section H.3 the recoverable signal's frequencies are N -heavy.

Lemma H.35 (Frequency estimation for one-cluster signal, Lemma 7.3 in Chen et al. (2016)) . For a sufficiently small constant ε 0 &gt; 0 , any f 0 ∈ [ -F, F ] , and ∆ 0 &gt; 0 , given an ( ε 0 , ∆ 0 ) -one-cluster signal z ( t ) around f 0 , Procedure FREQUENCYRECOVERY1CLUSTER , returns ˜ f 0 with | ˜ f 0 -f 0 | ≲ ∆ 0 · √ ∆ 0 T with probability at least 1 -2 -Ω( k ) . The following theorem shows the algorithm for ultra-high sensitivity frequency estimation.

Theorem H.36 (Ultra-high sensitivity frequency estimation algorithm with low success probability) . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t and x ( t ) = x ∗ ( t ) + g ( t ) be our observable signal where ∥ g ( t ) ∥ 2 T ≤ c ∥ x ∗ ( t ) ∥ 2 T for a sufficiently small constant c . Then Procedure FREQUENCYRECOVERYKCLUSTER returns a set L of O ( k/ ( ε 0 ε 1 ε 2 )) frequencies that cover all N 2 -heavy clusters and have high SNR (See Definition H.32) of x ∗ , which uses poly( k, ε -1 , ε -1 0 , ε -1 1 , ε -1 2 , log(1 /δ )) log( FT ) samples and poly( k, ε -1 , ε -1 0 , ε -1 1 , ε -1 2 , log(1 /δ )) log 2 ( FT ) time.

In particular, for ∆ 0 = ε -1 poly( k, log(1 /δ )) /T and N 2 2 := ε 1 ε 2 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) , with probability 0 . 9 , for any f ∗ with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

there exists an ˜ f ∈ L satisfying

Proof. By Lemma H.34 and Lemma H.35, we prove the theorem.

Theorem H.37 (Ultra-high sensitivity frequency estimation algorithm with high success probability) . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t and x ( t ) = x ∗ ( t ) + g ( t ) be our observable signal where ∥ g ( t ) ∥ 2 T ≤ c ∥ x ∗ ( t ) ∥ 2 T for a sufficiently small constant c . Then Procedure FREQUENCYRECOVERYKCLUSTER returns a set L of O ( k/ ( ε 0 ε 1 ε 2 )) frequencies that covers all N 2 -heavy clusters of x ∗ , which uses poly( k, ε -1 , ε -1 0 , ε -1 1 , ε -1 2 , log(1 /δ )) log( FT ) samples and poly( k, ε -1 , ε -1 0 , ε -1 1 , ε -1 2 , log(1 /δ )) log 2 ( FT ) time.

In particular, for ∆ 0 = ε -1 poly( k, log(1 /δ )) /T and N 2 2 := ε 1 ε 2 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) , with probability 1 -2 -Ω( k ) , for any f ∗ with there exists an ˜ f ∈ L satisfying

The following lemma shows the approximation error guarantee for the recoverable signal x S of the ultra-high sensitivity frequency estimation algorithm (Theorem H.37).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma H.38 (Recoverable signal's approximation error guarantee) . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t and x ( t ) = x ∗ ( t ) + g ( t ) be our observable signal. Let N 2 1 := ε 1 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) . Let C 1 , · · · , C l are the N 1 -heavy clusters from Definition H.11. Let S ∗ denotes the set of frequencies f ∗ ∈ { f j } j ∈ [ k ] such that, f ∗ ∈ C i for some i ∈ [ l ] , and

<!-- formula-not-decoded -->

Let S denotes the set of frequencies f ∗ ∈ S ∗ such that, f ∗ ∈ C j for some j ∈ [ l ] , and

Then, we have that,

<!-- formula-not-decoded -->

Proof. Following from the fact that √ 1 + ε = 1 + O ( ε ) for ε &lt; 1 , we have

We have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows from Corollary H.22, the third step follows from triangle inequality, the forth step follows from Claim H.12.

Thus, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows from the definition of g , the third step follows from Claim H.12.

Therefore,

<!-- formula-not-decoded -->

where the first step follows from Lemma H.23, the second step follows from triangle inequality, the third step follows from x S -x S ∗ being k -Fourier-sparse and Property VI of Lemma H.7, the forth step change the order of the terms, the fifth step follows from Claim H.12, the sixth step follows from ∥ H ( x -x S ) ∥ T + ∥ H ( x S -x S ∗ ) ∥ T ≤ √ 2 √ ∥ H ( x -x S ) ∥ 2 T + ∥ H ( x S -x S ∗ ) ∥ 2 T , the seventh step follows from Lemma H.21, the eighth step follows from Eq. (49), the ninth step follows from Eq. (50), the last step follows from ε = ε 0 = ε 1 = ε 2 .

The following lemma shows that the recoverable signal x S ( t ) 's energy is close to the observation signal x ( t ) .

<!-- formula-not-decoded -->

Lemma H.39 (Recoverable signal's energy) . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t and x ( t ) = x ∗ ( t ) + g ( t ) be our observable signal. Let N 2 1 := ε 1 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) . Let C 1 , · · · , C l are the N 1 -heavy clusters from Definition H.11. Let S ∗ denotes the set of frequencies f ∗ ∈ { f j } j ∈ [ k ] such that, f ∗ ∈ C i for some i ∈ [ l ] , and

Let S denotes the set of frequencies f ∗ ∈ S ∗ such that, f ∗ ∈ C j for some j ∈ [ l ] , and

Then, we have that,

Proof. We have that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows from Corollary H.22, the third step follows from triangle inequality, the forth step follows from Claim H.12.

## H.9 High SNR and recoverable signals

Lemma H.40 (High SNR and recoverable approximation error guarantee) . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t and x ( t ) = x ∗ ( t ) + g ( t ) be our observable signal. Let N 2 1 := ε 1 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) . Let C 1 , · · · , C l are the N 1 -heavy clusters from Definition H.11. Let S ∗ denotes the set of frequencies f ∗ ∈ { f j } j ∈ [ k ] such that, f ∗ ∈ C i for some i ∈ [ l ] , and

<!-- formula-not-decoded -->

Let S denotes the set of frequencies f ∗ ∈ S ∗ such that, f ∗ ∈ C j for some j ∈ [ l ] , and

<!-- formula-not-decoded -->

And S f is defined in Definition H.32. Then, we have that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We have that

And then for any f ∈ S \ S f , j = h σ,b ( f ) , we have that where the first step follows from Definition H.32, the second step is from simple algebra.

Let T = S \ S f . And for any j ∈ [ B ] , if j ∈ [ B ] \ S g , T j = { i ∈ S | h σ,b ( f i ) = j } . Otherwise, T j = ∅ . Moreover, we have that for any f ∈ supp( x T j ∗ H ) ,

<!-- formula-not-decoded -->

From Property VI of Lemma H.7, we have that

<!-- formula-not-decoded -->

By Lemma H.29, we know that ( x ∗ · H ) ∗ G ( j ) σ,b always satisfies Property II (in Definition H.27):

<!-- formula-not-decoded -->

where the first step follows from the definition of the norm, the second step is from Lemma H.29, the third step is due to Parseval's Theorem, the forth step is based on the Large Offset event not happening, the fifth step is based on simple algebra, the last step is because of Lemma H.29.

We also have that

<!-- formula-not-decoded -->

where the first step follows from Definition of T , the second step follows from Eq. (53), the third step is based on definition of norm, the forth step follows from simple algebra, the fifth step follows from Parseval's Theorem, the six step is due to Large Offset event not happening, the seventh step is due to Lemma H.29, the eighth step follows from Eq. (51).

In the following, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step is due to the definition of norm, the second step follows from g ( t ) = 0 when t / ∈ [0 , T ] , the third step follows from Parseval's Theorem, the forth step is because of Lemma H.29, the fifth step is from Parseval's Theorem, the sixth step is based on g ( t ) = 0 when t / ∈ [0 , T ] , the seventh step is from | H ( t ) | 2 ≤ 1 , the last step is from the definition of norm. We have that

<!-- formula-not-decoded -->

where the first step follows from Eq. (55), the second step follows from simple algebra, the third step is due to Eq.(54), the forth step is because of the reason that δ is much smaller than ε and ε &lt; 1 .

Lemma H.41 (High SNR signal's energy) . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t and x ( t ) = x ∗ ( t ) + g ( t ) be our observable signal. Let N 2 1 := ε 1 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) . Let C 1 , · · · , C l are the N 1 -heavy clusters from Definition H.11. Let S ∗ denotes the set of frequencies f ∗ ∈ { f j } j ∈ [ k ] such that, f ∗ ∈ C i for some i ∈ [ l ] , and

<!-- formula-not-decoded -->

Let S denotes the set of frequencies f ∗ ∈ S ∗ such that, f ∗ ∈ C j for some j ∈ [ l ] , and

<!-- formula-not-decoded -->

Let S f be defined in Definition H.32. Then, we have that,

<!-- formula-not-decoded -->

Proof. We have that,

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows from Corollary H.22, the third step follows from triangle inequality, the forth step follows from Claim H.12, where the last step follows from Lemma H.40.

## H.10 (3 + √ 2 + ε ) -approximate algorithm

In this section, we prove the main result: a (3 + √ 2+ ε ) -approximate Fourier interpolation algorithm, which significantly improves the accuracy of Chen et al. (2016)'s result.

<!-- formula-not-decoded -->

Theorem H.42 (Fourier interpolation with (3+ √ 2+ ε ) -approximation error) . Let x ( t ) = x ∗ ( t )+ g ( t ) , where x ∗ is k -Fourier-sparse signal with frequencies in [ -F, F ] . Given samples of x over [0 , T ] we can output y ( t ) such that with probability at least 1 -2 -Ω( k ) ,

Our algorithm uses poly( k, ε -1 , log(1 /δ )) log( FT ) samples and poly( k, ε -1 , log(1 /δ )) · log 2 ( FT ) time. The output y is poly( k, ε -1 , log(1 /δ )) -Fourier-sparse signal.

Proof. Let N 2 2 := ε 1 ε 2 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) , N 2 1 := ε 1 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) be the heavy cluster parameter.

First, by Lemma H.12, there is a set of frequencies S ∗ ⊂ [ k ] and x S ∗ ( t ) = v j e 2 π i f j t such that

<!-- formula-not-decoded -->

∑ j ∈ S ∗

Furthermore, each f j with j ∈ S ∗ belongs to an N 1 -heavy cluster C j with respect to the filter function H defined in Definition H.6.

By Definition H.11 of heavy cluster, it holds that

<!-- formula-not-decoded -->

By Definition H.11, we also have | C j | ≤ k · ∆ h , where ∆ h is the bandwidth of ̂ H . Let ∆ ∈ R + , and ∆ &gt; k · ∆ h , which implies that C j ⊆ [ f j -∆ , f j +∆] . Thus, we have

<!-- formula-not-decoded -->

By Corollary H.22, there is a set of frequencies S ⊂ S ∗ and x S ( t ) = ∑ j ∈ S v j e 2 π i f j t such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the following part, we will only focus on recovering the high SNR frequency. Let S f be defined in Definition H.32. It's to know S f ⊂ S By applying Theorem H.37, there is an algorithm that outputs a set of frequencies L ⊂ R such that, | L | = O ( k/ ( ε 0 ε 1 ε 2 )) , and with probability at least 1 -2 -Ω( k ) , for any f j with j ∈ S f , there is a ˜ f ∈ L such that,

<!-- formula-not-decoded -->

We define a map p : R → L as follows:

Then, x S ( t ) can be expressed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step follows from the definition of x S ( t ) , the last step follows from interchanging the summations.

<!-- formula-not-decoded -->

For each ˜ f i ∈ L , by Corollary H.2 with x ∗ = x S , ∆ = ∆ √ ∆ T , we have that there exist degree d = O ( T ∆ √ ∆ T + k 3 log k + k log 1 /δ ) polynomials P i ( t ) corresponding to f i ∈ L such that,

Define the following function family:

<!-- formula-not-decoded -->

By Claim H.16, for function family F , K Uniform[0 , T] = O (( | L | d ) 4 log 3 ( | L | d )) .

Note that ∑ ˜ f i ∈ L e 2 π i ˜ f i t P i ( t ) ∈ F .

By Lemma H.18, we have that, choosing a set W of O ( ε -1 K Uniform[0 , T] log( | L | d/ρ )) i.i.d. samples uniformly at random over duration [0 , T ] is a ( ε, ρ ) -WBSP.

By Lemma H.19, there is an algorithm that runs in O ( ε -1 | W | ( | L | d ) ω -1 log(1 /ρ )) -time using samples in W , and outputs y ′ ( t ) ∈ F such that, with probability 1 -ρ ,

Then by Lemma H.3, we have that there is a ( kd ) -Fourier-sparse signal y ( t ) , such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where δ ′ &gt; 0 is any positive real number, thus, y can be arbitrarily close to y ′ . Moreover, the sparsity of y ( t ) is

<!-- formula-not-decoded -->

Therefore, the total approximation error can be upper bounded as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By re-scaling ε and δ , we prove the theorem.

## I Improving Band-Limited Interpolation Precision in a Smaller Range

In this section, we show that the approximation error of the Fourier interpolation algorithm developed in Section H can be further improved, if we only care about the signal in a shorter time duration [0 , (1 -c ) T ] for c ∈ (0 , 1) . The main result of this section is Theorem I.4.

## I.1 Control noise

Lemma I.1. Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t and x ( t ) = x ∗ ( t ) + g ( t ) be our observable signal. Let N 2 1 := ε 1 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) . Let C 1 , · · · , C l are the N 1 -heavy clusters from Definition H.11. Let S ∗ denotes the set of frequencies f ∗ ∈ { f j } j ∈ [ k ] such that, f ∗ ∈ C i for some i ∈ [ l ] , and

<!-- formula-not-decoded -->

Let S denotes the set of frequencies f ∗ ∈ S ∗ such that, f ∗ ∈ C j for some j ∈ [ l ] , and

Then, we have that,

<!-- formula-not-decoded -->

Proof. Following from the fact that √ 1 + ε = 1 + O ( ε ) for ε &lt; 1 , we have

We have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows the definition of g , the third step follows from Claim H.12.

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows from for any function x : R → C , (1 -c ) ∥ x ∥ T ′ ≤ ∥ x ∥ T , the third step follows from Property I of Lemma H.7 and (1 -c ) / 2 &lt; ( 1 2 -2 s 1 ) s 3 , the forth step follows from Claim H.12, the fifth step follows from ∥ H ( x -x S ) ∥ T + ∥ H ( x S -x S ∗ ) ∥ T ≤ √ 2 √ ∥ H ( x -x S ) ∥ 2 T + ∥ H ( x S -x S ∗ ) ∥ 2 T , the sixth step follows from Lemma H.21, the seventh step follows from Eq. (49), the last step follows from ε = ε 0 = ε 1 = ε 2 .

Parameters setting By Section C.3 in Chen et al. (2016), we choose parameters for filter function ( H ( t ) , H ( f )) as follows:

- ∆ h is determined by the parameters of filter ( H ( t ) , ̂ H ( f )) in Eq. (20): ∆ h ≂ s 1 ℓ s 3 T . Combining the setting of s 1 , s 3 ℓ , we should set ∆ h ≥ max( ˜ O ( k 5 log(1 /δ )) / ( εT ) , O ( k log( k/δ ) / ( cT ))) .
- ̂ · By Eq. (19) in the proof of Property VI of filter function ( H ( t ) , ̂ H ( f )) , we need (1 -s 3 (1 -2 s 1 )) · ˜ O ( k 4 ) ≤ ε , thus we have that min( 1 1 -s 3 , s 1 ) ≥ ˜ O ( k 4 ) /ε . · In the proof of Property V of filter function ( H ( t ) , ̂ H ( f )) , we set ℓ ≳ k log( k/δ ) . · In the proof of Lemma I.1, we set (1 -c ) / 2 &lt; ( 1 2 -2 s 1 ) s 3 . Thus, we have that min( s 3 , 1 -4 s 1 ) ≥ 1 -c 2 or equivalently min( 1 1 -s 3 , s 1 / 4) ≥ 2 c .

## I.2 ( √ 2 + ε ) -approximation ratio

Corollary I.2 (Corollary of Theorem H.37) . Let x ∗ ( t ) = ∑ k j =1 v j e 2 π i f j t and x ( t ) = x ∗ ( t ) + g ( t ) be our observable signal where ∥ g ( t ) ∥ 2 T ≤ c 0 ∥ x ∗ ( t ) ∥ 2 T for a sufficiently small constant c 0 . Then Procedure FREQUENCYRECOVERYKCLUSTER returns a set L of O ( k/ ( ε 0 ε 1 ε 2 )) frequencies that covers all N 2 -heavy clusters of x ∗ , which uses poly( k, c -1 , ε -1 , ε -1 0 , ε -1 1 , ε -1 2 , log(1 /δ )) log( FT ) samples and poly( k, c -1 , ε -1 , ε -1 0 , ε -1 1 , ε -1 2 , log(1 /δ )) log 2 ( FT ) time.

In particular, for ∆ 0 = c -1 ε -1 poly( k, log(1 /δ )) /T and N 2 2 := ε 1 ε 2 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) , with probability 1 -2 -Ω( k ) , for any f ∗ with there exists an ˜ f ∈ L satisfying

Theorem I.4 ( ( √ 2 + ε ) -approximate Fourier interpolation algorithm with shrinking range) . Let x ( t ) = x ∗ ( t ) + g ( t ) , where x ∗ is k -Fourier-sparse signal with frequencies in [ -F, F ] . Let T ′ =

<!-- formula-not-decoded -->

Remark I.3. The proof is similar with the proof of Theorem H.37.

<!-- formula-not-decoded -->

T (1 -c ) . Given samples of x over [0 , T ] , we can output y ( t ) such that with probability at least 1 -2 -Ω( k ) ,

Our algorithm uses poly( k, ε -1 , c -1 , log(1 /δ )) log( FT ) samples and poly( k, ε -1 , c -1 , log(1 /δ )) · log 2 ( FT ) time. The output y is poly( k, ε -1 , c -1 , log(1 /δ )) -Fourier-sparse signal.

<!-- formula-not-decoded -->

Proof. Let N 2 1 := ε 1 ( ∥ g ( t ) ∥ 2 T + δ ∥ x ∗ ( t ) ∥ 2 T ) be the heavy cluster parameter.

<!-- formula-not-decoded -->

First, by Lemma H.12, there is a set of frequencies S ∗ ⊂ [ k ] and x S ∗ ( t ) = ∑ j ∈ S ∗ v j e 2 π i f j t such that

Furthermore, each f j with j ∈ S belongs to an N 1 -heavy cluster C j with respect to the filter function H defined in Definition H.6.

By Definition H.11 of heavy cluster, it holds that

<!-- formula-not-decoded -->

By Definition H.11, we also have | C j | ≤ k · ∆ h , where ∆ h is the bandwidth of ̂ H . Let ∆ ∈ R + , and ∆ &gt; k · ∆ h , which implies that C j ⊆ [ f j -∆ , f j +∆] . Thus, we have

By Corollary H.22, there is a set of frequencies S ⊂ S ∗ and x S ( t ) = ∑ j ∈ S v j e 2 π i f j t such that

Let g ′ = x -x S ∗ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now it is enough to recover only x S , instead of x ∗ .

By applying Theorem I.2, there is an algorithm that outputs a set of frequencies L ⊂ R such that, | L | = O ( k/ ( ε 0 ε 1 ε 2 )) , and with probability at least 1 -2 -Ω( k ) , for any f j with j ∈ S , there is a ˜ f ∈ L such that,

We define a map p : R → L as follows:

Then, x S ( t ) can be expressed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first step follows from the definition of x S ( t ) , the last step follows from interchanging the summations.

<!-- formula-not-decoded -->

For each ˜ f i ∈ L , by Corollary H.2 with x ∗ = x S , ∆ = ∆ √ ∆ T , we have that there exist degree d = O ( T ∆ √ ∆ T + k 3 log k + k log 1 /δ ) polynomials P i ( t ) corresponding to f i ∈ L such that,

Define the following function family:

<!-- formula-not-decoded -->

By Claim H.16, for function family F , K Uniform[cT / 2 , T(1 -c / 2)] = O (( | L | d ) 4 log 3 ( | L | d )) .

Note that ∑ ˜ f i ∈ L e 2 π i ˜ f i t P i ( t ) ∈ F .

By Lemma H.18, we have that, choosing a set W of O ( ε -1 K Uniform[cT / 2 , T(1 -c / 2)] log( | L | d/ρ )) i.i.d. samples uniformly at random over duration [0 , T ] is a ( ε, ρ ) -WBSP.

By Lemma H.19, there is an algorithm that runs in O ( ε -1 | W | ( | L | d ) ω -1 log(1 /ρ )) -time using samples in W , and outputs y ′ ( t ) ∈ F such that, with probability 1 -ρ ,

<!-- formula-not-decoded -->

Then by Lemma H.3, we have that there is a O ( kd ) -Fourier-sparse signal y ( t ) , such that

<!-- formula-not-decoded -->

Moreover, the sparsity of y ( t ) is kd = kO ( T ∆ √ ∆ T + k 3 log k + k log 1 /δ ) = poly( k, ε -1 , c -1 , log(1 /δ )) .

where δ ′ &gt; 0 is any positive real number. Thus, y can be arbitrarily close to y ′ .

Therefore, the total approximation error can be upper bounded as follows:

<!-- formula-not-decoded -->

where the first step follows from triangle inequality, the second step follows from Eq. (66), the third step follows from Eq. (65), the forth step follows from Triangle Inequality again, the fifth step follows from Eq. (64), the sixth step follows from Lemma I.1, the seventh step follows from Lemma H.39, and the last step is straightforward.

By re-scaling ε and δ , we prove the theorem.

## J Broader Impact

By cutting the approximation constant from ∼ 100 to 3 + √ 2 , our methods could materially shorten scan times in MRI, reduce power consumption in compressive sensing devices, and improve fidelity in spectrum-sparse communication systems, thus benefiting healthcare, environmental monitoring,

and data transmission. At the same time, higher-quality reconstructions from fewer samples may amplify surveillance capabilities or aid in generating convincingly doctored audio/video; responsible adoption therefore demands privacy safeguards, transparent validation on non-ideal data, and ethical oversight whenever the technology is applied to sensitive domains.
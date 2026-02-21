1

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

## Gradient-Free Approaches is a Key to an Efficient Interaction with Markovian Stochasticity

## Anonymous Author(s)

Affiliation Address email

## Abstract

This paper deals with stochastic optimization problems involving Markovian noise with a zero-order oracle. We present and analyze a novel derivativefree method for solving such problems in strongly convex smooth and non-smooth settings with both one-point and two-point feedback oracles. Using a randomized batching scheme, we show that when mixing time τ of the underlying noise sequence is less than the dimension of the problem d , the convergence estimates of our method do not depend on τ . This observation provides an efficient way to interact with Markovian stochasticity: instead of invoking the expensive first-order oracle, one should use the zero-order oracle. Finally, we complement our upper bounds with the corresponding lower bounds. This confirms the optimality of our results.

## 1 Introduction

Stochasticity is a fundamental aspect of many optimization problems, naturally arising in the field of machine learning [48, 28]. Stochastic gradient descent (SGD) [45] and its accelerated variants [39, 25] have become a de facto optimizers for modern large models training. Theoretical properties of SGD have been extensively studied under various statistical frameworks [36, 24, 10, 56], often relying on the assumption that noise is independent and identically distributed (i.i.d.). However, in many real-world applications - including reinforcement learning (RL) [6, 16], distributed optimization [35, 31], and bandit problems [3] - noise is not i.i.d., instead exhibiting correlations or Markovian structure . For instance, in the mentioned growing field of RL, sequential interactions with the environment induce state-dependent structure of the noise, creating a need for non-i.i.d. noise aware algorithms. Although several gradient-based methods for Markovian stochastic oracles have been studied in the past decade [14, 18], policy optimization in RL is based solely on reward feedback, making traditional methods inapplicable, since there is no access to first-order information [46, 9, 19]. Zero-order optimization (ZOO) methods are specifically developed to address such problems, and are used in scenarios where gradients are unavailable or prohibitively expensive to compute. Apart from RL, ZOO techniques are widely employed in adversarial attack generation [8], hyperparameter tuning [47, 57], continuous bandits [7, 49] and other applications [54, 33]. While the literature on ZOO is extensive, this work is, to our knowledge, the first study of optimization problem with both zero-order information and Markovian noise , aimed at developing an optimal algorithm for a large family of problems from the intersection of these two areas.

Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

## 1.1 Related works 34

/diamondmath Zero-order methods is one of the key and oldest areas of optimization. There are various 35 zero-order approaches, here we can briefly highlight, e.g., one-dimensional methods [32, 42] 36 or their high-dimensional analogues [41], ellipsoid algorithms [58] and searches along random 37 directions [4]. Currently, the most popular and most studied mechanism behind ZOO 38 methods is the finite-difference approximation of the gradient described in [43, 20, 40]. The 39 idea is simple: querying two sufficiently close points is essentially equivalent to finding a 40 value of the directional derivative of the function: 41

<!-- formula-not-decoded -->

where e is a random direction. It can be a random coordinate, a vector from the Euclidean 42 sphere or a sample of the Gaussian distribution. The approximation (1) in turn leads back to 43 the gradient methods or coordinate algorithms of Nesterov [38]. There are, however, several 44 differences: 45

· First, to get full gradient information, the algorithm would need d queries instead of one 46 gradient oracle call (here d is the dimension of x ). 47

48

49

· Second, if the ZO oracle is inexact, i.e. only noisy values of function are available, then finite difference schemes can fail if noise components do not cancel out.

50

51

52

53

54

The setting of the second point, when function evaluations experience zero-mean additive perturbations, is called Stochastic ZOO . The stochasticity, as noted before, is abundant in the modern optimization world. To tackle this issue, additional assumptions about the noise structure are required. Here we briefly discuss two main ideas adopted in the literature, and refer the reader to Section 2 for precise definitions.

In the case of two-point feedback , we assume that for a fixed value of the noise variable one 55 can call the stochastic zero-order oracle at least twice. It means that we can compute the 56 finite difference approximation of the following form: 57

<!-- formula-not-decoded -->

Such approximation produces an estimate for the directional derivative of a noisy realization 58 f ( · , ξ ) of the function f . As mentioned before, the approximation (2) can be used instead of 59 the (stochastic) gradient in first-order methods. In the case of independent randomness, a 60 large number of works are based on this idea. There are results for both non-smooth and 61 smooth convex problems built on classical and accelerated gradient methods of Nesterov and 62 Spokoiny [40]. In the scope of our paper, we are interested in the results for smooth strongly 63 convex problems from [17], namely estimates on zero-order oracle calls to achieve ε -solution 64 in terms of ‖ x -x ∗ ‖ : O ( dσ 2 2 µ 2 ε ). Here σ 2 is introduced as the variance of the gradient, i.e. it is 65 assumed that E ξ ∇ f ( x, ξ ) = ∇ f ( x ) and E ξ ‖∇ f ( x, ξ ) -∇ f ( x ) ‖ 2 ≤ σ 2 2 . The main limitation 66 of two-point approach is that several evaluations with the same noise variable are required, 67 which is well suited for problems like empirical risk optimization [34], but can be a major 68 barrier for RL or online optimization. 69 In the one-point feedback setting, a more general stochasticity is assumed. In this case, each 70 call to the zero-order oracle generates a new randomness. Now the approximation (1) looks 71 as follows 72

73

74

75

76

77

78

79

80

Using different

ξ

+

<!-- formula-not-decoded -->

ξ

and

-

Instead, it is assumed that in (3) renders any conditions on the properties of

E

ξ

f

(

x, ξ

) =

f

(

x

) and

E

ξ

|

f

(

x, ξ

t

feedback, the major problem is choosing the right shift leads to a poor gradient estimate.

)

-

f

(

x

)

|

2

(

) useless.

∇

f

·

, ξ

2

for the finite difference scheme.

. With one-point

≤

σ

1

Picking it too small results in an amplification of the additive noise, and taking

t

too big methods with one-point approximation is worse than for two-point feedback. In particular,

Because of this variance trade-off, the optimal rate for for smooth strongly convex problems we have the following estimate on zero-order oracle

calls [23]:

2

2

O

1

σ

3

ε

2

(

d

µ

).

Although zero-order gradient approximation schemes suffer from high variance, there is a 81 surprising property that makes them superior in non-smooth optimization [22, 44, 49]. The 82 idea goes back to the 70s and utilizes the fact that 83

<!-- formula-not-decoded -->

√

In fact, it can be shown that f t is dG t -smooth if f is G -Lipschitz. This makes zero-order 84 approximation a suitable candidate for a stochastic gradient of f t . Optimizing this function 85 with a first-order method produces some solution, but it may not be the optima of f [22]. 86 From this point, there is a game - for small t the functions f and f t are closer and for big t 87 the function f t is easier to optimize as it gets smoother. 88

89

90

91

92

93

94

In more recent works, there have been many improvements in theoretical understanding of ZO methods. The authors consider higher-order smoothness of the underlying function [2], tackle non-convex non-smooth problems [44], take arbitrary Bregman geometry to benefit in terms of oracle complexity [49, 29], and come up with sharp information-theoretic lower bounds to understand computational limits [15, 1]. But none of them consider Markovian stochasticity.

95

96

97

98

99

100

101

102

103

104

105

106

107

108

109

110

111

112

113

114

115

116

117

118

119

120

121

122

123

124

125

126

127

128

129

/diamondmath Markovian first-order methods. While the literature on stochastic optimization with i.i.d. noise is extensive, research addressing the Markovian setting remains relatively sparse. In our paper, we focus on the most "friendly" type of uniformly geometrically ergodic Markov chains (see Section 2 for precise definitions).

Duchi et al. [14] conducted pioneering work on non-i.i.d. noise, investigating the Ergodic Mirror Descent algorithm and establishing optimal convergence rates for non-smooth convex problems. For smooth problems there were different attempts to get record-breaking estimates on the first-order oracle [12, 11, 59, 18]. Finally, the optimal results were obtained for both convex and non-convex problems in the works of Beznosikov et al. [5], Solodkin et al. [52]. In particular, for smooth strongly convex objectives under Markovian noise the authors give the complexity of the form: O ( τσ 2 2 µ 2 ε ), where τ is defined as the mixing time of the corresponding Markov chain (see Section 2). Note that these works utilize Multilevel Monte Carlo (MLMC) batching technique, which helps to effectively interact with Markovian noise. We will need this approach as well. Note that it was first considered in Markovian gradient optimization by Dorfman and Levy [13] for automatic adaptation to unknown τ .

/diamondmath Hypothesis. The complexity estimate for strongly convex first-order stochastic methods is O ( σ 2 2 µ 2 ε ) [36, 37]. Lower bounds for the same class of problems and methods show that the result is unimprovable [58]. As mentioned before, the transition from i.i.d. stochasticity to Markovian stochasticity increases the estimate by τ times. This result is also optimal as shown by Beznosikov et al. [5]. At the same time, going from gradient oracle to zero-order methods adds a multiplier d in the two-point feedback and d 2 / ε in the one-point case. And this estimate is unimprovable as well [1, 15]. The hypothesis arises that the transition to zero- at once:

order Markov optimization adds two multipliers dτ

and

d

2

τ

illustrated in the following diagram for two-point

/

ε

for two- and one-point. It is feedback:

## 1.2 Our contribution

Our main contribution is the answer to the hypothesis above: surprisingly, it is not true . In more detail:

<!-- image -->

/diamondmath Accelerated SGD. We present the first analysis of Zero-Order Accelerated SGD under Markovian noise, considering both two-point and one-point feedback. Contrary to the expected multiplicative scaling of convergence rates with both dimensionality and mixing time, our analysis reveals a significant acceleration, as presented in Figure 1. It turns out that if τ is smaller than d , our results do not differ at all from the gradient-free methods

130

131

132

133

134

135

136

137

138

139

140

141

142

143

144

145

146

147

148

149

150

151

152

153

154

155

156

157

158

159

160

161

with independent stochasticity. The key technique behind this acceleration is described in Section 2.1. The theory is also numerically validated in Section 3.

/diamondmath Non-smooth problems. We also consider non-smooth problems with Markovian noise. Using the smoothing technique we come up with a corresponding upper bounds in this case, as shown in Figure 1. The details of these bounds are presented in Appendix A.

| Smooth   | Smooth                  | Smooth                     | Non-smooth IID             | Non-smooth IID                |
|----------|-------------------------|----------------------------|----------------------------|-------------------------------|
|          | IID                     | Markov.                    |                            | Markov.                       |
| FO       | σ 2 2 µ 2 ε [45]        | τ σ 2 2 µ 2 ε [5]          | G 2 µ 2 ε [50]             | τ G 2 µ 2 ε [14] 1            |
| ZO 2P    | d σ 2 2 µ 2 ε [30]      | ( d + τ ) σ 2 2 µ 2 ε      | d G 2 µ 2 ε [22]           | ( d + τ ) G 2 µ 2 ε           |
| ZO 1P    | d 2 σ 2 1 µ 3 ε 2 [2] 2 | d ( d + τ ) Lσ 2 1 µ 3 ε 2 | d 2 σ 2 1 G 2 µ 4 ε 3 [23] | d ( d + τ ) σ 2 1 G 2 µ 4 ε 3 |

/diamondmath Computational efficiency. First, as noted above, our method gives the same oracle complexity for any τ ≤ d . Moreover, if we assume that calling a zero-order oracle is d times cheaper than computing the corresponding gradient, then the gradient method with Markov noise will require resources proportionally to d · τ - the cost of one oracle call is d and the complexity scales as τ for the first-order method from Figure 1. At the same time, the resource complexity of our zero-order method is proportional to d + τ .

/diamondmath Lower bounds. In Section 2.3 we establish the first information-theoretic lower bounds for solving Markovian optimization problems with one-point and two-point feedback. Our results match the convergence guarantee of our algorithm up to logarithmic factors, showing that the analysis is accurate and no further improvement is possible.

Table 1: Notations &amp; Definitions

<!-- formula-not-decoded -->

| Sym.                                                                                                   | Definition                                                                                                                                                                                                                                                                                                                                                                                                           | Sym.                                      | Definition                                                                                                                                                                                                                                                                |
|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ‖·‖ , 〈· , ·〉 Z , Z Q P ξ , E ξ { Z k } RB d 2 ,RS d 2 e a n ≲ b n a n /similarequal b n T = ˜ O ( S ) | Norm, dot product, assumed Euclidean by default Complete separable metric space, its Borel σ -algebra Markov kernel on Z ×Z Probability, Expectation under initial distribution ξ 3 Canonical process with kernel Q Uniform distribution on unit a /lscript 2 -ball, -sphere Random direction, e ∼ RS d 2 ∃ c ∈ R (problem-independent): a n ≤ cb n for all n a n ≲ b n and b n ≲ a n T ≤ poly (log S ) · S as ε → 0 | ε d L µ G σ 2 1 σ 2 2 τ g , ˆ g f t ( x ) | ‖ x - x ∗ ‖ 2 Problem dimension Gradient's Lipshitz constant Strong convexity constant Function's Lipshitz constant &#124; F ( x,Z ) - f ( x ) &#124; 2 ≤ σ 2 1 ‖∇ F ( x,Z ) -∇ f ( x ) ‖ 2 ≤ σ 2 2 Mixing time of Z Gradient estimators E r [ f ( x + tr )] , r ∼ RB d 2 |

## 2 Main results

We are now ready for a more formal presentation. In this paper, we study the minimization problem where π is an unknown distribution and access to the function f (not to its gradient ∇ f ) is available through a stochastic one-point or two-point oracle F ( x, Z ).

In our analysis, we will use a set of assumptions on the underlying function f and its oracle, starting with smoothness and convexity:

Assumption 1. The function f is L -smooth on R d with L &gt; 0 , i.e., it is differentiable and there is a constant L &gt; 0 such that the following inequality holds for all x, y ∈ R d :

<!-- formula-not-decoded -->

In the two-point feedback setting, we require the following generalization:

Assumption 1 ′ . For all Z ∈ Z the function F ( · , Z ) is L -smooth on R d .

Note that the uniform 1 ′ implies 1.

1 The authors consider general convex case. Using standard restart technique, we get the corresponding bound in the strongly convex case.

2 The noise is assumed to be point-independent.

3 By construction, for any A ∈ Z , we have P ξ ( Z k ∈ A | Z k -1 ) = Q( Z k -1 , A ) , P ξ -a.s.

162

163

164

165

166

167

168

169

170

171

172

173

174

175

176

177

178

179

180

181

182

183

184

185

186

187

188

189

190

191

192

193

194

195

196

Assumption 2. The function f is µ -strongly convex on R d , i.e., it is continuously differentiable and there is a constant µ &gt; 0 such that the following inequality holds for all x, y ∈ R d :

<!-- formula-not-decoded -->

We now turn to assumptions on the sequence of noise states { Z i } ∞ i =0 . Specifically, we consider the case where { Z i } ∞ i =0 forms a time-homogeneous Markov chain. Let Q denote the corresponding Markov kernel. We impose the following assumption on Q to characterize its mixing properties:

Assumption 3. { Z i } ∞ i =0 is a stationary Markov chain on ( Z , Z ) with Markov kernel Q and unique invariant distribution π . Moreover, Q is uniformly geometrically ergodic with mixing time τ ∈ N , i.e., for every k ∈ N ,

<!-- formula-not-decoded -->

Assumption 3 is common in the literature on Markovian stochasticity [14, 12, 13, 5, 52]. It includes, for instance, irreducible aperiodic finite Markov chains [18]. The mixing time τ reflects how quickly the distribution of the chain approaches stationarity, providing a natural measure of the temporal dependence in the data.

Next, we specify our assumptions on the oracle. As discussed in Section 1.1, these assumptions differ based on the type of feedback.

Assumption 4 (for one-point) . For all x ∈ R d it holds that E π [ F ( x, Z )] = f ( x ) . Moreover, for all Z ∈ Z and x ∈ R d it holds that

<!-- formula-not-decoded -->

Assumption 4 ′ (for two-point) . For all x ∈ R d it holds that E π [ ∇ F ( x, Z )] = ∇ f ( x ) . Moreover, for all Z ∈ Z and x ∈ R d it holds that

<!-- formula-not-decoded -->

Recent works on stochastic ZOO methods have considered milder assumptions, such as bounded variance (see Section 1.1). However, the uniform boundedness assumed in Assumptions 4 and 4 ′ , is standard in analyses under Markovian noise [14, 12, 13, 5, 52]. These assumptions can be relaxed under stronger conditions, e.g., uniform convexity and smoothness of F ( · , Z ) [18].

Assumptions 3 and 4 allow us to reduce the variance of the noise via batching, similarly the to i.i.d. setting. This is captured in the following technical lemma:

Lemma 1. Let Assumptions 3 and 4(4 ′ ) hold. Then for any n ≥ 1 and x ∈ R d and any initial distribution ξ on ( Z , Z ) , we have

<!-- formula-not-decoded -->

## 2.1 Batching technique

In this section, we describe the main tools used to establish the ( d + τ )-type scaling of the error rate. We will focus on reducing the variance and bias of gradient estimators using a specialized batching approach.

We begin by fixing a common building block of our gradient estimators at a point x for both one-point and two-point feedback, as introduced in Section 1.1:

<!-- formula-not-decoded -->

These estimators exhibit a twofold randomness that affects how rapidly they concentrate 197 around the true gradient, as we will discuss below. 198

For clarity, we focus our discussion on the one-point case, although our conclusions extend 199 to the two-point case as well. 200

A widely used variance reduction technique is mini-batching , where one computes F ( x, Z i ) 201 over a batch of noise variables { Z i } n i =1 . The mini-batch gradient estimator is given by: 202

<!-- formula-not-decoded -->

Let us estimate the scaling of its variance E e E Z ‖ ˆ g mb -∇ f ‖ 2 with the noise level σ 2 1 . As E Z ˆ g mb ≈ d f ( x + te ) -f ( x -te ) 2 t ≈ d 〈∇ f, e 〉 we would like to estimate the following for any fixed direction e :

With that, we bound the variance:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Can the mini-batching scheme be improved?

This subsection explores an unexpected source of improvement that contradicts our initial hypothesis. Specifically, we identify an inefficiency in the current use of samples Z i , which becomes evident from two perspectives. Equation (8) shows the variance scales as τ n . If we could reduce τ by a factor of k , we would need k -times fewer samples to maintain the same variance. This leads us to the idea of sparsified sampling. We partition the Markov noise chain { Z i } into k subchains { Z k · i + r } for r = 0 . . . k -1. This corresponds to a mixing time of /ceilingleft τ k /ceilingright for each subchain (see (3)), effectively reducing temporal correlation - a natural consequence of sampling every k -th element of the original chain. Thus, sampling from any single subchain could yield a min( k, τ )-fold reduction in the number of samples needed (although such procedure would still require all intermediate oracle calls, yielding no computational speedup).

For a concrete illustration of that inefficiency, consider a lazy Markov chain that remains in the same state for (an average of) τ steps before transitioning uniformly at random. In such a case, all oracle queries F ( x, Z ) for a fixed x return the same value for τ consecutive steps. Therefore, retaining only every τ -th estimate ˆ g would yield a mini-batch of equivalent quality.

In summary, we observe that the mini-batching scheme could, in principle, operate just as effectively by retaining only every k -th sample and discarding the rest. This might suggest that better utilization of the samples is possible. First order methods, nevertheless, are unable to exploit this redundancy (as shown by [5]'s lower bound) and are effectively forced to wait out the τ -step mixing window. In contrast, we can exploit this structure by querying finite differences along different directions to estimate the gradient better. Specifically, we construct d subchains, and use the sample from the r -th subchain Z d · i + r to estimate r -th partial derivative F ( x + te r ,Z ) -F ( x -te r ,Z ) 2 t , effectively restoring the full gradient coordinate-wise.

Let us estimate the resulting variance reduction. First, we achieve a d -fold reduction by reconstructing all d gradient coordinates. Second, each coordinate now operates on a chain with mixing time /ceilingleft τ d /ceilingright , yielding an additional factor of min( d, τ ). However, because batches are now split across d coordinates, each batch is d times smaller than before, introducing a factor of d loss. The net variance reduction is therefore min( d, τ ), and the final scaling becomes d · dτ min( d,τ ) = d · max( d, τ ) /similarequal d ( d + τ ).

## Random directions

203

204

205

206

207

208

209

210

211

212

213

214

215

216

217

218

219

220

221

222

223

224

225

226

227

228

229

230

231

232

233

234

235

236

237

238

This insight can be extended to a simpler yet equally effective method. Instead of assigning 239 directions deterministically, we associate each sample with a random direction e ∈ RS d 2 , 240 forming the estimator: 241

<!-- formula-not-decoded -->

While the above discussion was intuitive, we now outline a more formal approach (see 242 Lemma 5 for details). As lazy Markov chain is effectively equivalent to stochastic i.i.d. 243 τ -point feedback setting, we follow Corollary 2 of [15], who decompose the total variance 244 into two terms: 245

<!-- formula-not-decoded -->

Each of the two terms individually eliminates one factor from the d 2 τ dependence. 246

247

248

249

250

251

252

253

254

255

256

257

258

259

260

261

262

263

The first term:

<!-- formula-not-decoded -->

is independent of τ since Assumption 4 bounds each term directly.

For the second term, we observe that E e ˆ g rd = E e ˆ g mb , and thus the bound involves E ∥ ∥ E e ˆ g mb -∇ f ( x ) 2 ∥ ∥ . This is crucially different from the d 2 τ dependence that appeared in the mini-batch case, when we considered E ∥ ∥ ˆ g mb -∇ f ( x ) 2 ∥ ∥ . Intuitively, the expectation over directions helps recover the full gradient rather than a directional component, thereby reducing variance with respect to d .

## Multilevel Monte Carlo

The estimator ˆ g rd is not our final construction. While it controls variance, the temporal correlation in noise may introduce significant bias. A well-established approach to mitigating this is MLMC, widely used in the statistical literature [27, 26], and more recently in gradient optimization [13, 5]. Here is our interpretation.

With parameters J, l, M, B from Table 2, { Z i } - 2 J l samples from Z and { e i } - random directions we introduce MLMC estimator:

<!-- formula-not-decoded -->

ˆ g ml is our final gradient estimator, with the following guarantees:

Lemma 2 (for one-point) . Let Assumptions 1, 3 and 4 hold. For any initial distribution 1 ξ on ( Z , Z ) the gradient estimates ˆ g ml satisfy E [ˆ g ml ] = E [ ˆ g rd [ 2 /floorleft log 2 M /floorright l ]] . Moreover,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

One can note that although ˆ g ml requires, on average, E [ 2 J lB ] = log 2 2 M · B oracle calls, the 264 variance is only reduced by a factor of B . In contrast, the bias is reduced significantly - by a 265 factor of BM . 266

267

268

269

270

271

272

273

## 2.2 Algorithm

We now present the full version of Algorithm 1, which incorporates the gradient estimators discussed in the previous section and uses a slightly modified variant of Nesterov's Accelerated Gradient Descent at its core.

While technically we prove four separate upper bounds covering both one- and two-point feedback under smooth and non-smooth assumptions, they follow the same scheme which we will illustrate in the one-point smooth case.

1 Note that ˆ g ml (specifically Z 1 ) indirectly depends on the chain's initial distribution. As our algorithm is going to repeatedly call ˆ g ml , next iteration's initial distribution is current iteration's final distribution. This fact makes the estimates correlated. We sidestep this problem by assuming any initial distribution.

274

275

276

277

278

279

280

281

282

283

284

285

286

287

288

289

290

291

292

293

294

295

296

297

298

299

300

301

302

303

304

Table 2: Parameters of Algorithm 1

| Hyperparameters   | Hyperparameters         | Momentums Batch hidden   | Momentums Batch hidden   | parameters                               |
|-------------------|-------------------------|--------------------------|--------------------------|------------------------------------------|
| γ                 | Stepsize, ∈ (0; 3 4 L ] | β                        | √ 4 p 2 µγ 3             | Batch size. If 2 J l>M , then 0          |
| t                 | Approximation step      | η                        | 3 β 2 pµγ = √ 3 µγ       | Random, J ∼ Geom( 1 / 2 )                |
| B                 | Batch size multiplier   | θ                        | pη - 1 - 1 βpη - 1 - 1   | Batch size limit, M = 1 p + 2 β          |
| N                 | Number of iterations    | p                        | See Appendix             | ( /floorleft log 2 M /floorright +1) · B |

Lemma 4 establishes key properties of the smoothed objective function. Lemma 5 provides bounds on the bias and variance of the baseline estimator ˆ g rd . Lemma 2 then quantifies how the MLMC scheme amplifies or reduces these statistics. Finally, in Section C.4, we combine the results of these lemmas to prove the first part of Theorem 1, bounding Algorithm 1's error. By tuning the parameters appropriately, we obtain the following iteration complexity bound:

## Algorithm 1 Randomized Accelerated ZO GD

| 1:   | Initialization: x 0 f = x 0 ; see Table 2.                                   |
|------|------------------------------------------------------------------------------|
| 2:   | for k = 0 , 1 , 2 ,...,N - 1 do                                              |
| 3:   | x k g = θx k f +(1 - θ ) x k                                                 |
| 4:   | Sample J k , { e i } , { F ( x k g ± te i ,Z ( ± ) i ) }                     |
| 5:   | Calculate ˆ g k = ˆ g ml ( x ) k +1 k k                                      |
| 6:   | x f = x g - pγ ˆ g                                                           |
| 7:   | x k +1 = ηx k +1 f +( p - η ) x k f + +(1 - p )(1 - β ) x k +(1 - p ) βx k g |

Theorem 1. Let Assumptions 1 to 4 hold, and consider problem (4) solved by Algorithm 1. Then, for any target accuracy ε and batch size multiplier B (see Tables 1 and 2 for notation), and for a suitable choice of γ, t, p , the number of oracle calls required to ensure E ‖ x N -x ∗ ‖ 2 ≤ ε is bounded by

<!-- formula-not-decoded -->

Theorem 1 ′ . Let Assumptions 1 ′ to 4 ′ hold, and consider problem (4) solved by Algorithm 1. Then, for any target accuracy ε and batch size multiplier B (see Tables 1 and 2 for notation), and for a suitable choice of γ, t, p , the number of oracle calls required to ensure E ‖ x N -x ∗ ‖ 2 ≤ ε is bounded by

<!-- formula-not-decoded -->

Remark. The iteration complexity of the algorithm, i.e., the number of iterates x k generated (equal to the oracle complexity divided by B ), is bound by ˜ O (√ L µ log 1 ε ) as the batch size multiplier B goes to infinity. This matches the optimal convergence rates for optimization with exact gradients [39].

## 2.3 Lower bounds

Here we present theorems demonstrating that no algorithm can asymptotically outperform Algorithm 1 in the smooth, strongly convex setting with either one- or two-point feedback.

Theorem 2. (Lower bounds) For any (possibly randomized) algorithm that solves the problem (4) , there exists a function f that satisfies Assumptions 1 to 4 (1 ′ to 4 ′ ), s.t. in order to achieve ε -approximate solution in expectation E ‖ x N -x ∗ ‖ 2 ≤ ε , the algorithm needs at least

<!-- formula-not-decoded -->

Remark. These results assume bounded second moments rather than uniform noise bounds. 305 We explain how to adapt them to our setting, incurring only logarithmic overheads, in 306 Section E.2. 307

Discussion. We now compare our results to existing work. Akhavan et al. [2] analyze a 308 special case of the one-point setting where the noise is independent of the query points. This 309

310

311

312

313

314

315

316

317

318

319

320

321

322

323

324

325

326

327

328

329

330

331

332

333

334

335

336

337

338

339

340

341

342

343

344

345

346

aligns with our one-point oracle model and allows i.i.d. sampling as a Markov chain with fixed mixing time τ = 1. The only factor they do not consider is σ 2 1 , which, however, appears in their proof with additional µ 2 factor if used with scaled Gaussian noise. We discuss this further in Appendix E.

In the work of Beznosikov et al. [5], a first-order Markovian oracle is considered, but the hard instance problem is a one-dimensional quadratic function, which makes first-order and zero-order information equivalent. Their result therefore corresponds to the d = 1 case in the two-point regime. Duchi et al. [15] provide tight lower bounds for general convex functions under two-point feedback. Their techniques can be extended to the strongly convex case by incorporating a shared quadratic component across the hard instances, as detailed in Appendix E, Theorem 10, yielding the bound we state for the two-point oracle with τ = 1.

Our novel contribution lies in establishing a lower bound that scales as dτ in the onepoint regime for large τ ; see Theorem 9. While our analysis relies on classical tools such as multidimensional hypothesis testing, the Markovian structure requires new bound on distances between joint distributions and the use of clipping. Detailed proofs, discussions, and further remarks on clipping appear in Appendix E.

## 3 Experiments

This section empirically supports our theoretical convergence rates and lower bounds, with particular focus on the stochastic component where we claim linear scaling in d + τ instead of dτ .

Setup. Our setup repeats the problem we used to prove the lower bounds (see Appendix E and [51]). We consider a quadratic objective f ( x ) = 1 2 ‖ x ‖ 2 and a two-point Markovian oracle F ( x, Z ) = f ( x ) + 〈 x, Z 〉 . The noise sequence { Z i } is a lazily updated standard Gaussian vector with variance σ 2 2 . Figure 2 illustrates how the optimization error of Algorithm 1 scales with mixing time, problem dimension, and different values of σ 2 2 .

Figure 2: Optimization error ε = ‖ x N -x ∗ ‖ 2 after N = 10 3 iterations. Starting point error ‖ x 0 -x ∗ ‖ 2 = 10 -2 . Stepsize γ = 10 -3 , t = 10 -5 . The results are averaged over 10 4 runs.

<!-- image -->

Discussion. The results confirm the linear dependence of the error on both the problem dimension d and the mixing time τ . The noise parameter σ 2 controls the influence of the stochastic part. In Fig. (a), where σ 2 2 = 10 -3 , the stochastic component dominates, while in Fig. (c), with σ 2 2 = 10 -5 , it is negligible. Fig. (b) shows an intermediate regime that smoothly interpolates between the two, yet maintains the linear scaling. The deterministic part (c) shows no dependence on mixing time, but grows linearly with d , which aligns with our theory (Theorem 1 ′ ). The stochastic part (a) scales as ( d + τ ), also matching the bound from the Theorem 1 ′ .

## References

- [1] Arya Akhavan, Massimiliano Pontil, and Alexandre Tsybakov. Exploiting higher order smoothness in derivative-free optimization and continuous bandits. Advances in Neural Information Processing Systems , 33:9017-9027, 2020.

- [2] Arya Akhavan, Evgenii Chzhen, Massimiliano Pontil, and Alexandre B Tsybakov. 347 Gradient-free optimization of highly smooth functions: improved analysis and a new 348 algorithm. Journal of Machine Learning Research , 25(370):1-50, 2024. 349
- [3] Peter Auer. Finite-time analysis of the multiarmed bandit problem. Machine Learning , 350 47:235-256, 2002. 351

352

353

354

- [4] El Houcine Bergou, Eduard Gorbunov, and Peter Richtárik. Stochastic three points method for unconstrained smooth minimization. SIAM Journal on Optimization , 30(4): 2726-2749, 2020.

355

356

357

358

359

360

361

362

363

364

365

366

367

368

369

370

371

372

373

374

375

376

377

378

379

380

381

382

383

384

385

386

387

388

389

390

391

- [5] Aleksandr Beznosikov, Sergey Samsonov, Marina Sheshukova, Alexander Gasnikov, Alexey Naumov, and Eric Moulines. First order methods with markovian noise: from acceleration to variational inequalities. Advances in Neural Information Processing Systems , 36, 2024.
- [6] Jalaj Bhandari, Daniel Russo, and Raghav Singal. A finite time analysis of temporal difference learning with linear function approximation. In Conference on learning theory , pages 1691-1692. PMLR, 2018.
- [7] Sébastien Bubeck, Nicolo Cesa-Bianchi, et al. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends® in Machine Learning , 5(1):1-122, 2012.
- [8] Pin-Yu Chen, Huan Zhang, Yash Sharma, Jinfeng Yi, and Cho-Jui Hsieh. Zoo: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models. In Proceedings of the 10th ACM workshop on artificial intelligence and security , pages 15-26, 2017.
- [9] Krzysztof Choromanski, Mark Rowland, Vikas Sindhwani, Richard Turner, and Adrian Weller. Structured evolution with compact architectures for scalable policy optimization. In International Conference on Machine Learning , pages 970-978. PMLR, 2018.
- [10] Aymeric Dieuleveut, Nicolas Flammarion, and Francis Bach. Harder, better, faster, stronger convergence rates for least-squares regression. Journal of Machine Learning Research , 18(101):1-51, 2017.
- [11] Thinh T Doan. Finite-time analysis of markov gradient descent. IEEE Transactions on Automatic Control , 68(4):2140-2153, 2022.
- [12] Thinh T Doan, Lam M Nguyen, Nhan H Pham, and Justin Romberg. Convergence rates of accelerated markov gradient descent with applications in reinforcement learning. arXiv preprint arXiv:2002.02873 , 2020.
- [13] Ron Dorfman and Kfir Yehuda Levy. Adapting to mixing time in stochastic optimization with markovian data. In International Conference on Machine Learning , pages 54295446. PMLR, 2022.
- [14] John C Duchi, Alekh Agarwal, Mikael Johansson, and Michael I Jordan. Ergodic mirror descent. SIAM Journal on Optimization , 22(4):1549-1578, 2012.
- [15] John C Duchi, Michael I Jordan, Martin J Wainwright, and Andre Wibisono. Optimal rates for zero-order convex optimization: The power of two function evaluations. IEEE Transactions on Information Theory , 61(5):2788-2806, 2015.
- [16] Alain Durmus, Eric Moulines, Alexey Naumov, Sergey Samsonov, and Hoi-To Wai. On the stability of random matrix product with markovian noise: Application to linear stochastic approximation and td learning. In Conference on Learning Theory , pages 1711-1752. PMLR, 2021.
- [17] Pavel Dvurechensky, Eduard Gorbunov, and Alexander Gasnikov. An accelerated 392 directional derivative method for smooth stochastic convex optimization. European 393 Journal of Operational Research , 290(2):601-621, 2021. 394

395

396

397

398

399

400

401

402

403

404

405

406

407

408

409

410

411

412

413

414

415

416

417

418

419

420

421

422

423

424

425

426

427

428

429

430

431

432

433

434

435

436

437

438

439

440

441

442

443

- [18] Mathieu Even. Stochastic gradient descent under markovian sampling schemes. In International Conference on Machine Learning , pages 9412-9439. PMLR, 2023.
- [19] Maryam Fazel, Rong Ge, Sham Kakade, and Mehran Mesbahi. Global convergence of policy gradient methods for the linear quadratic regulator. In International conference on machine learning , pages 1467-1476. PMLR, 2018.
- [20] Abraham D. Flaxman, Adam Tauman Kalai, and H. Brendan McMahan. Online convex optimization in the bandit setting: gradient descent without a gradient. In Proceedings of the Sixteenth Annual ACM-SIAM Symposium on Discrete Algorithms , SODA '05, page 385-394, USA, 2005. Society for Industrial and Applied Mathematics. ISBN 0898715857.
- [21] Alexander Gasnikov, Darina Dvinskikh, Pavel Dvurechensky, Eduard Gorbunov, Aleksandr Beznosikov, and Alexander Lobanov. Randomized Gradient-Free Methods in Convex Optimization , pages 1-15. Springer International Publishing, Cham, 2020. ISBN 978-3-030-54621-2. doi: 10.1007/978-3-030-54621-2\_859-1. URL https://doi.org/10.1007/978-3-030-54621-2\_859-1 .
- [22] Alexander Gasnikov, Anton Novitskii, Vasilii Novitskii, Farshed Abdukhakimov, Dmitry Kamzolov, Aleksandr Beznosikov, Martin Takac, Pavel Dvurechensky, and Bin Gu. The power of first-order smooth optimization for black-box non-smooth problems. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 7241-7265. PMLR, 17-23 Jul 2022. URL https://proceedings.mlr.press/v162/gasnikov22a.html .
- [23] Alexander V Gasnikov, Ekaterina A Krymova, Anastasia A Lagunovskaya, Ilnura N Usmanova, and Fedor A Fedorenko. Stochastic online optimization. single-point and multi-point non-linear multi-armed bandits. convex and strongly-convex case. Automation and remote control , 78:224-234, 2017.
- [24] Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM journal on optimization , 23(4):2341-2368, 2013.
- [25] Saeed Ghadimi and Guanghui Lan. Accelerated gradient methods for nonconvex nonlinear and stochastic programming. Mathematical Programming , 156(1):59-99, 2016.
- [26] Michael B. Giles. Multilevel monte carlo path simulation. Operations Research , 56(3): 607-617, 2008. doi: 10.1287/opre.1070.0496. URL https://doi.org/10.1287/opre. 1070.0496 .
- [27] Peter W. Glynn and Chang-Han Rhee. Exact estimation for markov chain equilibrium expectations. Journal of Applied Probability , 51A:377-389, 2014. ISSN 00219002. URL http://www.jstor.org/stable/43284129 .
- [28] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning . MIT Press, 2016. http://www.deeplearningbook.org .
- [29] Eduard Gorbunov, Pavel Dvurechensky, and Alexander Gasnikov. An accelerated method for derivative-free smooth stochastic convex optimization. SIAM Journal on Optimization , 32(2):1210-1238, 2022. doi: 10.1137/19M1259225. URL https: //doi.org/10.1137/19M1259225 .
- [30] Elad Hazan and Satyen Kale. Beyond the regret minimization barrier: Optimal algorithms for stochastic strongly-convex optimization. Journal of Machine Learning Research , 15(71):2489-2512, 2014. URL http://jmlr.org/papers/v15/hazan14a.html .
- [31] Bjorn Johansson, Maben Rabi, and Mikael Johansson. A simple peer-to-peer algorithm for distributed optimization in sensor networks. In 2007 46th IEEE Conference on Decision and Control , pages 4705-4710, 2007. doi: 10.1109/CDC.2007.4434888.

444

445

446

447

448

449

450

451

452

453

454

455

456

457

458

459

460

461

462

463

464

465

466

467

468

469

470

471

472

473

474

475

476

477

478

479

480

481

482

- [32] J. Kiefer. Sequential minimax search for a maximum. Proceedings of the American Mathematical Society , 4(3):502-506, 1953. ISSN 00029939, 10886826. URL http: //www.jstor.org/stable/2032161 .
- [33] Xiangru Lian, Yijun Huang, Yuncheng Li, and Ji Liu. Asynchronous parallel stochastic gradient for nonconvex optimization. Advances in neural information processing systems , 28, 2015.
- [34] Sijia Liu, Bhavya Kailkhura, Pin-Yu Chen, Paishun Ting, Shiyu Chang, and Lisa Amini. Zeroth-order stochastic variance reduction for nonconvex optimization. Advances in Neural Information Processing Systems , 31, 2018.
- [35] Cassio G. Lopes and Ali H. Sayed. Incremental adaptive strategies over distributed networks. IEEE Transactions on Signal Processing , 55(8):4064-4077, 2007. doi: 10. 1109/TSP.2007.896034.
- [36] Eric Moulines and Francis Bach. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. In J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems , volume 24. Curran Associates, Inc., 2011. URL https://proceedings.neurips.cc/ paper\_files/paper/2011/file/40008b9a5380fcacce3976bf7c08af5b-Paper.pdf .
- [37] Deanna Needell, Rachel Ward, and Nati Srebro. Stochastic gradient descent, weighted sampling, and the randomized kaczmarz algorithm. Advances in neural information processing systems , 27, 2014.
- [38] Yu Nesterov. Efficiency of coordinate descent methods on huge-scale optimization problems. SIAM Journal on Optimization , 22(2):341-362, 2012.
- [39] Yurii Nesterov. A method for solving the convex programming problem with convergence rate o (1/k2). In Doklad nauk Sssr , volume 269, page 543, 1983.
- [40] Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics , 17(2):527-566, 2017.
- [41] Donald J Newman. Location of the maximum on unimodal surfaces. Journal of the ACM (JACM) , 12(3):395-398, 1965.
- [42] J. Nocedal and S. Wright. Numerical Optimization . Springer Series in Operations Research and Financial Engineering. Springer New York, 2006. ISBN 9780387227429. URL https://books.google.ru/books?id=7wDpBwAAQBAJ .
- [43] Boris Polyak. Introduction to Optimization . Optimization Software - Inc., Publications Division, 1987.
- [44] Yuyang Qiu, Uday Shanbhag, and Farzad Yousefian. Zeroth-order methods for nondifferentiable, nonconvex, and hierarchical federated optimization. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems , volume 36, pages 3425-3438. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/file/0a70c9cd8179fe6f8f6135fafa2a8798-Paper-Conference.pdf .
- [45] Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals 483 of mathematical statistics , pages 400-407, 1951. 484

485

486

487

- [46] Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. arXiv preprint arXiv:1703.03864 , 2017.
- [47] Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P. Adams, and Nando de Freitas. 488 Taking the human out of the loop: A review of bayesian optimization. Proceedings of 489 the IEEE , 104(1):148-175, 2016. doi: 10.1109/JPROC.2015.2494218. 490

- [48] Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From 491 theory to algorithms . Cambridge university press, 2014. 492

493

494

- [49] Ohad Shamir. An optimal algorithm for bandit and zero-order convex optimization with two-point feedback. Journal of Machine Learning Research , 18(52):1-11, 2017.

495

496

497

498

499

500

501

502

503

504

505

506

507

508

509

510

511

512

513

514

515

516

517

518

519

520

521

522

523

- [50] Ohad Shamir and Tong Zhang. Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. In Sanjoy Dasgupta and David McAllester, editors, Proceedings of the 30th International Conference on Machine Learning , volume 28 of Proceedings of Machine Learning Research , pages 71-79, Atlanta, Georgia, USA, 17-19 Jun 2013. PMLR. URL https://proceedings.mlr.press/v28/ shamir13.html .
- [51] Alexander Shapiro, Darinka Dentcheva, and Andrzej Ruszczyński. Lectures on Stochastic Programming . Society for Industrial and Applied Mathematics, 2009. doi: 10.1137/1.9780898718751. URL https://epubs.siam.org/doi/abs/10.1137/ 1.9780898718751 .
- [52] Vladimir Solodkin, Andrew Veprikov, and Aleksandr Beznosikov. Methods for optimization problems with markovian stochasticity and non-euclidean geometry. arXiv preprint arXiv:2408.01848 , 2024.
- [53] Sebastian U. Stich. Unified optimal analysis of the (stochastic) gradient method, 2019. URL https://arxiv.org/abs/1907.04232 .
- [54] Ben Taskar, Vassil Chatalbashev, Daphne Koller, and Carlos Guestrin. Learning structured prediction models: A large margin approach. In Proceedings of the 22nd international conference on Machine learning , pages 896-903, 2005.
- [55] Alexandre B. Tsybakov. Lower bounds on the minimax risk , pages 77-135. Springer New York, New York, NY, 2009. ISBN 978-0-387-79052-7. doi: 10.1007/978-0-387-79052-7\_2. URL https://doi.org/10.1007/978-0-387-79052-7\_2 .
- [56] Sharan Vaswani, Francis Bach, and Mark Schmidt. Fast and faster convergence of sgd for over-parameterized models and an accelerated perceptron. In The 22nd international conference on artificial intelligence and statistics , pages 1195-1204. PMLR, 2019.
- [57] Jian Wu, Saul Toscano-Palmerin, Peter I Frazier, and Andrew Gordon Wilson. Practical multi-fidelity bayesian optimization for hyperparameter tuning. In Uncertainty in Artificial Intelligence , pages 788-798. PMLR, 2020.
- [58] David B Yudin and Arkadi S Nemirovskii. Informational complexity and efficient methods for the solution of convex extremal problems. Matekon , 13(2):22-45, 1976.
- [59] Yawei Zhao. Markov chain mirror descent on data federation. arXiv preprint 524 arXiv:2309.14775 , 2023. 525

927

928

929

930

931

932

933

934

935

936

937

938

939

940

941

942

943

944

945

946

947

948

949

950

951

952

953

954

955

956

957

958

959

960

961

962

963

964

965

966

967

968

969

970

971

972

973

974

975

976

977

978

979

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: main contributions of this paper are described accurately in a dedicated subsection (Section 1.2) of the introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: assumptions we use to prove the main results are presented in Section 2. The motivation for these assumptions as well their limitations are also described there.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: all assumptions and definitions are carefully stated. The complete proofs appear in the supplemental material and are properly referenced in the main part.

## Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Theorems and Lemmas that the proof relies upon should be properly referenced.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: see Section 3. The setup is fully disclosed.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

1034

1035

1036

1037

1038

1039

1040

1041

1042

1043

1044

1045

1046

1047

1048

1049

1050

1051

1052

1053

1054

1055

1056

1057

1058

1059

1060

1061

1062

1063

1064

1065

1066

1067

1068

1069

1070

1071

1072

1073

1074

1075

1076

1077

1078

1079

1080

1081

1082

1083

1084

- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No]

Justification: our experiments are rather a practical confirmation of theoretical results, and these experiments can be easily reproduced.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- Please see the NeurIPS code and data submission guidelines ( https://nips. cc/public/guides/CodeSubmissionPolicy ) for more details.
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: see Section 3, all parameters are described there.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The full details can be provided either with the code, in appendix, or as supplemental material.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

1085

1086

1087

1088

1089

1090

1091

1092

1093

1094

1095

1096

1097

1098

1099

1100

1101

1102

1103

1104

1105

1106

1107

1108

1109

1110

1111

1112

1113

1114

1115

1116

1117

1118

1119

1120

1121

1122

1123

1124

1125

1126

1127

1128

1129

1130

1131

1132

1133

1134

1135

1136

Justification: we use experiments to verify the theoretical rates and have no statistical effects associated with running the experiments.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- The assumptions made should be given (e.g., Normally distributed errors).
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No]

Justification: the experiments performed are not computationally heavy and can be reproduced on an average machine in a fairly reasonable amount of time.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: the research follows the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

1137

1138

1139

1140

1141

1142

1143

1144

1145

1146

1147

1148

1149

1150

1151

1152

1153

1154

1155

1156

1157

1158

1159

1160

1161

1162

1163

1164

1165

1166

1167

1168

1169

1170

1171

1172

1173

1174

1175

1176

1177

1178

1179

1180

1181

1182

1183

1184

1185

1186

1187

1188

1189

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: there is no societal impact of the work performed - we only develop the theoretical understanding of Optimization.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: the paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

1190

1191

1192

1193

1194

1195

1196

1197

1198

1199

1200

1201

1202

1203

1204

1205

1206

1207

1208

1209

1210

1211

1212

1213

1214

1215

1216

1217

1218

1219

1220

1221

1222

1223

1224

1225

1226

1227

1228

1229

1230

1231

1232

1233

1234

1235

1236

1237

1238

1239

Justification: the paper does not use existing assets.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The authors should cite the original paper that produced the code package or dataset.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/ datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: the paper does not propose new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: the paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects

1240

1241

1242

1243

1244

1245

1246

1247

1248

1249

1250

1251

1252

1253

1254

1255

1256

1257

1258

1259

1260

1261

1262

1263

1264

1265

1266

1267

1268

1269

1270

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: the paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs were used only for editing.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
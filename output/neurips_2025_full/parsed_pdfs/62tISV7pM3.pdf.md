19

## Corner Gradient Descent

## Anonymous Author(s)

Affiliation Address email

## Abstract

We consider SGD-type optimization on infinite-dimensional quadratic problems with power law spectral conditions. It is well-known that on such problems deterministic GD has loss convergence rates L t = O ( t -ζ ) , which can be improved to L t = O ( t -2 ζ ) by using Heavy Ball with a non-stationary Jacobi-based schedule (and the latter rate is optimal among fixed schedules). However, in the mini-batch Stochastic GD setting, the sampling noise causes the Jacobi HB to diverge; accordingly no O ( t -2 ζ ) algorithm is known. In this paper we show that rates up to O ( t -2 ζ ) can be achieved by a generalized stationary SGD with infinite memory. We start by identifying generalized (S)GD algorithms with contours in the complex plane. We then show that contours that have a corner with external angle θπ accelerate the plain GD rate O ( t -ζ ) to O ( t -θζ ) . For deterministic GD, increasing θ allows to achieve rates arbitrarily close to O ( t -2 ζ ) . However, in Stochastic GD, increasing θ also amplifies the sampling noise, so in general θ needs to be optimized by balancing the acceleration and noise effects. We prove that the optimal rate is given by θ max = min(2 , ν, 2 ζ +1 /ν ) , where ν, ζ are the exponents appearing in the capacity and source spectral conditions. Furthermore, using fast rational approximations of the power functions, we show that ideal corner algorithms can be efficiently approximated by practical finite-memory algorithms.

## 1 Introduction

It is well-known that Gradient Descent (GD) on quadratic problems can be accelerated using the 20 additional momentum term (the 'Heavy Ball' algorithm, [19]). For ill-conditioned problem, Heavy 21 Ball with a suitable non-stationary ('Jacobi') predefined schedule allows to accelerate a power-law 22 loss converge rate O ( t -ζ ) to O ( t -2 ζ ) , i.e. double the exponent ζ [8, 16]. This acceleration is the 23 best possible for non-adaptive schedules. 24

- On the other hand, for mini-batch Stochastic Gradient Descent (SGD) typically used in modern 25
- machine learning, the convergence rate picture is much more complicated, and much less is known 26 about possible acceleration. The natural quadratic problem in this case is the fitting of a linear model 27 with a sampled quadratic loss. In the power-law spectral setting, it was found in [4] that plain SGD 28 has two distinct convergent phases: either the sampling noise is weak and the SGD rate is the same 29 O ( t -ζ ) as for GD, or the convergence is slower due to the prevalence of the sampling noise. We 30 refer to these two scenarios as signaland noise-dominated , respectively. 31
- This picture was refined in several other works [18, 23, 24, 25, 29]. In particular, [29] examined 32 generalized SGDs with finite linear memory of any size (generalizing the momentum and similar 33 terms) and proved that with stationary schedules they all have the same phase diagram as plain SGD 34 -ζ
- (Figure 2 left); in particular, they do not accelerate the plain GD/SGD rate O ( t ) . 35
- On the other hand, the non-stationary Jacobi Heavy Ball accelerating deterministic GD from O ( t -ζ ) 36
- to O ( t -2 ζ ) fails for mini-batch Stochastic GD: it eventually starts to diverge due to the accumulating 37

sampling noise. [23] have proposed a non-stationary modification of SGD that achieves a quadratic 38 acceleration, but only on finite-dimensional problems. [29] have proposed a non-stationary modifi39 cation of the Heavy Ball/momentum algorithm that is heuristically expected (but not yet proved) to 40 achieve rates O ( t -θζ ) with some 1 &lt; θ &lt; 2 on infinite-dimensional problems. 41

To sum up, the topic of SGD acceleration in ill-conditioned quadratic problems is far from settled. 42

43

44

45

In the present paper we propose an entirely new approach to acceleration of (S)GD that both provides a new general geometric viewpoint and proves that, in a certain rigorous sense, SGD in the signaldominated regime can be accelerated from O ( t -ζ ) to O ( t -θζ ) with θ up to 2.

46

47

48

49

50

51

52

53

54

55

56

57

58

59

60

61

62

63

64

65

66

67

68

69

70

71

72

73

74

75

76

77

## Our contributions:

1. A view of generalized (S)GD as contours (Section 3). We show that stationary (S)GD algorithms with an arbitrary-sized linear memory can be identified with contours in the complex plane. This identification leverages the characteristic polynomials χ and the loss expansions of memoryM (S)GD from [29]. We show that all the information needed to compute the loss evolution is contained in a response map Ψ : { z ∈ C : | z | ≥ 1 } → C associated with χ . The map Ψ gives rise to the contour Ψ( { z ∈ C : | z | = 1 } ) and, conversely, can be reconstructed, along with the algorithm, from a given contour.
2. Corner algorithms (Section 4). A crucial role is played by contours that have a corner with external angle θπ, 1 &lt; θ &lt; 2 . We prove that the respective algorithms accelerate the plain GD rate O ( t -ζ ) to O ( t -θζ ) . However, in Stochastic GD such algorithms have the negative effect of amplifying the sampling noise. By balancing these two effects, we establish the precise phase diagram of feasible accelerations of SGD under power-law spectral assumptions (Figure 1 right). In particular, we identify three natural sub-phases in the signal-dominated phase; in one of them acceleration up to O ( t -2 ζ ) is theoretically feasible.
3. Implementation of Corner (S)GD (Section 5). Ideal corner algorithms require an infinite memory, but can be fast approximated by finite-memory algorithms using fast rational approximations of the power function z θ . Experiments with a synthetic problem and MNIST confirm the practical acceleration.

## 2 Background

This section is largely based on the paper [29] to which we refer for details.

Gradient descent with memory. Suppose that we wish to minimize a loss function L ( w ) on a linear space H . We consider gradient descent with sizeM memory that can be written as

<!-- formula-not-decoded -->

The vector w t is the current stept approximation to an optimal vector w ∗ , and u t is an auxiliary vector representing the 'memory' of the optimizer. These auxiliary vectors have the form u = ( u (1) , . . . u ( M ) ) T with u ( m ) ∈ H and can be viewed as sizeM columns with each component belonging to H . We refer to M as the memory size . The parameter α (learning rate) is scalar, the parameters b , c are M -dimensional column vectors, and D is a M × M scalar matrix. The algorithm can be viewed as a sequence of transformations of size-( M + 1) column vectors ( w t u t ) with H -valued components. Throughout the paper, we only consider stationary algorithms, in the sense that the parameters α, b , c , D do not depend on t. The simplest nontrivial special case of GD with memory is Heavy Ball [19], in which M = 1 and u t is the momentum.

Our theoretical results will rely on the assumption that L is quadratic: 78

<!-- formula-not-decoded -->

with a strictly positive definite H . Throughout the paper, we will mostly be interested in infinite79 dimensional Hilbert spaces H , and we slightly abuse notation by interpreting w T as the co-vector 80 (linear functional ⟨ w , ·⟩ ) associated with vector w . We will assume that H has a discrete spectrum 81 with ordered strictly positive eigenvalues λ k ↘ 0 . 82

Let w ∗ be the optimal value of L such that ∇ L ( w ∗ ) = Hw ∗ -q = 0 , and denote ∆ w t = w t -w ∗ . 83 Then, if ∆ w t and u t are eigenvectors of H with eigenvalue λ, then 84

<!-- formula-not-decoded -->

and the new vectors ∆ w t +1 , u t +1 are again eigenvectors of H with eigenvalue λ . As a result, 85 performing the spectral decomposition of ∆ w t , u t reduces the original dynamics (1) acting in H⊗ 86 R M +1 to a λ -indexed collection of independent dynamics each acting in R M +1 . 87

For quadratic L , evolution (1) admits an equivalent representation 88

<!-- formula-not-decoded -->

with constants ( p m ) M m =0 , ( q m ) M m =0 such that ∑ M m =0 p m = 1 . These constants are found from the 89 characteristic polynomial 90

<!-- formula-not-decoded -->

91

92

93

94

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

Batch SGD with memory. In batch Stochastic Gradient Descent (SGD), it is assumed that the loss has the form L ( w ) = E x ∼ ρ ℓ ( x , w ) , where ρ is some probability distribution of data points x and ℓ ( x , w ) is the loss at the point x . In the algorithm (1), we replace ∇ L by ∇ L B t , where B t is a random batch of | B | points sampled from distribution ρ , and ∇ L B is the empirical approximation to L , i.e. L B ( w ) = 1 | B | ∑ x ∈ B ℓ ( x , w ) . The samples B t at different steps t are independent.

We assume ℓ to have the quadratic form ℓ ( x , w ) = 1 2 ( x T w -y ( x )) 2 for some scalar target function y ( x ) . Here, the inner product x T w can be viewed as a linear model acting on the feature vector x . By projecting to the subspace of linear functions, we can assume w.l.o.g. that the target function y ( x ) is itself linear in x , i.e. f ( x ) = x T w ∗ with some optimal parameter vector w ∗ . (Later we will slightly weaken this assumption to also cover unfeasible solutions w ∗ . ) Then the full loss is quadratic as in Eq. (2): L ( w ) = E x ∼ ρ 1 2 ( x T ∆ w ) 2 = 1 2 ∆ w T H ∆ w , where ∆ w = w -w ∗ and the Hessian H = E x ∼ ρ [ xx T ] .

Mean loss evolution, SE approximation, and the propagator expansion. Since the trajectory w t in SGD is random, it is convenient to study the deterministic trajectory of batch-averaged losses L t = E B 1 ,...,B t -1 L ( w t ) . The sequence L t can be described exactly in terms of the second moments of w t , u t that admit exact evolution equations. An important aspect of this evolution is that it involves 4'th order moments of the data distribution ρ and so cannot in general be solved using only the second-order information available in the Hessian H = E x ∼ ρ [ xx T ] .

A convenient approach to handle this difficulty is the Spectrally-Expressible (SE) approximation proposed in [25]. It consists in assuming that there exist constants τ 1 , τ 2 such that for all positive definite operators C in H

In fact, this approximation holds exactly for some natural types of distribution ρ (translationinvariant, gaussian). Otherwise, if the r.h.s. is only an upper or lower bound for the l.h.s., this implies a respective relation between the actual losses and the losses computed under the SE approximation. Theoretical predictions obtained under assumption (6) show good quantitative agreement with experiment on real data. We refer to [25, 29] for further discussion of the SE approximation.

<!-- formula-not-decoded -->

The main benefit of the SE approximation is that it allows to write a convenient loss expansion

<!-- formula-not-decoded -->

with scalar noise propagators U t and signal propagators V t . The signal propagators describe the 118 error reduction during optimization in the absence of sampling noise, while the noise propagators 119 describe the perturbing effect of sampling noise injected at times t 1 , . . . , t m . 120

For our main results in Sections 3, 4, we will assume that τ 2 = 0 , implying particularly simple 121 formulas for U t , V t : 122

<!-- formula-not-decoded -->

123

124

125

126

127

128

129

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

where e k is a normalized eigenvector for λ k , and it is also assumed that optimization starts from w 0 = 0 so that ∆ w 0 = w 0 -w ∗ = -w ∗ .

Importantly, the batch size | B | affects L t only through the denominator in the coefficient in U t . The deterministic GD corresponds to the limit | B |→∞ : in this limit U t ≡ 0 and L t = 1 2 V t +1 .

Convergence/divergence regimes. Given expansion (7), we can deduce various convergence properties of the loss from the properties of the propagators V t , U t .

Theorem 1 ([29]) . Let numbers L t be given by expansion (7) with some U t ≥ 0 , V t ≥ 0 . Let U Σ = ∑ ∞ t =1 U t and V Σ = ∑ ∞ t =1 V t .

1. [Convergence] Suppose that U Σ &lt; 1 . At t →∞ , if V t = O (1) (respectively, V t = o (1) ), then also L t = O (1) (respectively, L t = o (1) ).
2. [Divergence] If U Σ &gt; 1 and V t &gt; 0 for at least one t , then sup t =1 , 2 ,... L t = ∞ .
3. [Signal-dominated regime] Suppose that there exist constants ξ V , C V &gt; 0 such that V t = C V t -ξ V (1 + o (1)) as t → ∞ . Suppose also that U Σ &lt; 1 and U t = O ( t -ξ U ) with some ξ U &gt; max( ξ V , 1) . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

4. [Noise-dominated regime] Suppose that there exist constants ξ V &gt; ξ U &gt; 1 , C U &gt; 0 such that U t = C U t -ξ U (1 + o (1)) and V t = O ( t -ξ V ) as t →∞ . Let also that U Σ &lt; 1 . Then

Spectral power laws. The detailed convergence results in items 3, 4 of Theorem 1 require us to know the asymptotics of the propagators U t , V t . To this end we introduce power-law spectral assumptions on the eigenvalues and eigencomponents of w ∗ in our optimization problem:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with some constants Λ , Q &gt; 0 and exponents ν &gt; 0 , ζ &gt; 0 . Such power laws are common in kernel methods or overparameterized models, and can be derived theoretically or observed empirically [1, 2, 3, 7, 10, 26, 27]. Conditions (11), (12) (or their weaker, inequality forms) are usually referred to as the capacity and source conditions, respectively [9]. The exponent ζ is akin to an inverse effective condition number: lower ζ means that the target and the solution have a heavier spectral tail of eigencomponents with small λ , making the problem harder. The exponent ν is akin to an inverse effective dimensionality of the problem: lower ν means a larger number of eigenvectors above a given spectral parameter λ . Only the source condition (12) matters for the non-stochastic GDrates, but in SGD the capacity condition (11) also becomes important due to the sampling noise.

If 0 &lt; ζ &lt; 1 , then the source condition (12) is inconsistent with w ∗ having a finite H -norm, i.e., strictly speaking, w ∗ is not an element of H . Such a solution is called unfeasible . In fact, unfeasible scenarios are quite common both theoretically and in practice (see Section F). The Corner SGD to be proposed in Section 4 will be especially suitable for unfeasible scenarios. Note also that if ν &lt; 1 2 , then U 1 = ∞ and so L t ≡ ∞ , i.e. the loss immediately diverges.

Stability and asymptotics of the propagators. Let us say that a square matrix A is strictly stable if all its eigenvalues are less than 1 in absolute value. It is natural to require the matrices S λ to be strictly stable for all λ ∈ spec( H ) , since otherwise U t , V t , and hence L t , will not generally even converge to 0 as t →∞ . At λ = 0 the matrix S λ =0 has eigenvalue 1 and additionally the eigenvalues of the matrix D ; accordingly, we will assume that D is strictly stable.

161

162

163

Figure 1: Left: The phase diagram of stationary finite-memory SGD from [25, 29]. Right: Maximum acceleration factor θ max = min(2 , ν, 2 ζ +1 /ν ) for Corner SGD in the signal-dominated regime (see Theorem 4).

<!-- image -->

Theorem 2 ([29]) . Suppose that D and S λ are strictly stable for all λ ∈ spec( H ) . Recalling the characteristic polynomial χ ( µ, λ ) = det( µ -S λ ) = P ( µ ) -λQ ( µ ) , define the effective learning rate

<!-- formula-not-decoded -->

and assume that α eff &gt; 0 . Then, under spectral assumptions (11) , (12) with ν &gt; 1 2 , the propagators 164 V t , U t given by Eq. (8) obey, as t →∞ , 165

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combined with Theorem 1, this result yields the ( ζ, 1 /ν ) -phase diagram shown in Figure 1 left. In 166 particular, the region ν &gt; 1 , 0 &lt; ζ &lt; 2 -1 /ν represents the signal-dominated phase in which 167 the noise effects are relatively weak and the loss convergence L t ∝ t -ζ has the same exponent ζ 168 as plain deterministic GD. This holds for all stationary finiteM algorithms and so such algorithms 169 cannot accelerate the exponent. In the present paper we will focus on the signal-dominated phase 170 and propose an 'infinite-memory' generalization of SGD that does accelerate the exponent. 171

172

173

174

175

176

## 3 The contour view of generalized (S)GD

We consider the propagator expansion (7) as a basis for our arguments. Observe that we can write the expression ( 1 0 T ) S t λ ( -α c ) appearing in the definition of propagator U t in Eq. (8) as

<!-- formula-not-decoded -->

where | µ | = r is a contour in the complex plane encircling all the eigenvalues of S λ . Next, simple calculation (see Section A) shows that

<!-- formula-not-decoded -->

where P ( µ ) -λQ ( µ ) is the characteristic polynomial of S λ introduced in Eq. (5), and 177

<!-- formula-not-decoded -->

We see, in particular, that the propagators U t depend on the algorithm parameters only through the 178 function Ψ : 179

<!-- formula-not-decoded -->

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

197

198

199

200

201

202

203

204

205

206

207

208

209

210

Figure 2: Left: The map Ψ = P Q for Heavy Ball with P ( µ ) = ( µ -1)( µ -0 . 4) and Q ( µ ) = -µ . The contour γ = Ψ( { µ : | µ | = 1 } ) encircles spec( H ) . The map Ψ bijectively maps {| µ | &gt; 1 } to the exterior open domain D γ with boundary γ . See Sec. B for more examples and a general discussion of memory-1 contours. Right: Contour γ corresponding to a corner map Ψ with angle θπ .

<!-- image -->

A similar observation can also be made regarding the propagators V t . Indeed, V t 's are different 180 from U t 's in that they involve the expression ( 1 0 T ) S t λ ( 1 0 ) instead of ( 1 0 T ) S t λ ( -α c ) . The contour 181 representation for ( 1 0 T ) S t λ ( 1 0 ) is similar to Eq. (16), and then a simple calculation gives 182

<!-- formula-not-decoded -->

As a result,

<!-- formula-not-decoded -->

Recall from Eqs. (4),(5) that P can be any monic polynomial (i.e., with leading coefficient 1) of degree M + 1 such that P (1) = 0 , while Q can be any polynomials of degree not greater than M . Since by Eq. (7) the loss trajectory L t is completely determined by the propagators U t , V t , we see that designing a stationary SGD with memory is essentially equivalent to designing a rational function Ψ subject to these simple conditions. By (4), the function Ψ = P Q can be interpreted as describing the (frequency) response of the gradient sequence ( ∇ L ( w t )) to the sequence ( w t ) .

̸

Let us consider the map Ψ from the stability perspective. Recall that we expect S λ k to be strictly stable for all the eigenvalues λ k ∈ spec( H ) . In terms of Ψ = P Q this means that Ψ( µ ) = λ k for all µ ∈ C such that | µ | ≥ 1 . This shows, in particular, that we can set the radius r = 1 in Eqs. (19), (21). Additionally, if D is strictly stable, then S 0 has only one simple eigenvalue of unit absolute value, µ = 1 , and so Ψ( µ ) = 0 for | µ | = 1 , µ = 1 . Let us introduce the curve γ as the image of the unit circle under the map Ψ . Then the last condition means that the curve γ goes through the point 0 only once, at µ = 1 .

̸

̸

In general, the curve γ can have a complicated shape with self-intersections, and the map Ψ may not be injective on the domain | µ | ≥ 1 . In particular, the singularity of Ψ at µ = ∞ is ∝ µ M +1 -deg( Q ) , so in a vicinity of µ = ∞ the function Ψ is injective if and only if deg( Q ) = M (and in general Ψ may also have other singularities at | µ | &gt; 1 ). However, we may expect natural, non-degenerate algorithms to correspond to simple non-intersecting curves γ and injective maps Ψ on | µ | ≥ 1 . For example, this is the case for plain (S)GD and Heavy Ball, where γ is a circle and an ellipse, respectively (Fig. 2 left). See Section B for a general discussion of memory-1 algorithms.

Given a non-intersecting (Jordan) contour γ, denote by D γ the respective exterior open domain. Then, by Riemann mapping theorem, there exists a bijective holomorphic map Ψ γ : { µ ∈ C : | µ | &gt; 1 } → D γ . Additionally, by Carathéodory's theorem 1 (see e.g. [11], p. 13) this map extends continuously to the boundary, Ψ γ : { µ ∈ C : | µ | = 1 } → γ . Such maps Ψ γ are non-unique, forming a three-parameter family Ψ γ ◦ f, where f is a conformal automorphism of { µ ∈ C : | µ | &gt; 1 } . However, recall that our maps Ψ = P Q had the properties Ψ( ∞ ) = ∞ and Ψ(1) = 0 . These two requirements for Ψ γ uniquely fix the conformal isomorphism and hence Ψ γ .

1 Carathéodory's theorem considers bounded domains, but our domains { µ ∈ C : | µ | &gt; 1 } and D γ are conformally isomorphic to bounded ones by simple transformations z = 1 / ( µ -µ 0 ) .

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

This suggests the following reformulation of the design problem for stationary SGD with memory. Rather than starting with the algorithm in the matrix or sequential forms (1), (4), we start with a contour γ or the associated Riemann map Ψ γ , and ensure a fast decay of the respective propagators U t , V t given by (19), (21) (and hence, by Theorem 1, of the loss L t ). Of course, the resulting map Ψ γ will not be rational in general, but we can subsequently approximate it with a rational function P Q and in this way approximately reconstruct the algorithm.

## 4 Corner algorithms

To motivate the algorithms introduced in this section, observe from Eqs. (9), (14) that in the signaldominated regime of stationary memoryM SGD, we can decrease the coefficient C L in the asymptotic formula L t = (1 + o (1)) C L t -ζ by increasing α eff while keeping the total noise coefficient U Σ &lt; 1 . Since Ψ(1) = 0 , α eff can be reformulated in terms of Ψ as

<!-- formula-not-decoded -->

Thus, increasing α eff means making -d Ψ dµ (1) a possibly smaller positive number. Regarding U Σ = 222 ∑ ∞ t =1 U t , note first that, by (19), it can be written as 223

<!-- formula-not-decoded -->

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

239

240

241

242

243

244

245

Indeed, since the function (Ψ( µ ) -λ ) -1 is holomorphic in {| µ | &gt; 1 } and vanishes as µ →∞ , the integrals ∮ here vanish for all nonpositive integers t = 0 , -1 , -2 , . . . so that ∑ t collapses to the squared L 2 norm by Parseval's identity. If the resulting series (23) converges, we can always ensure U Σ &lt; 1 by making the batch size | B | large enough.

<!-- formula-not-decoded -->

It is then natural to try Ψ = Ψ γ with a contour γ having a corner at 0 with a particular angle. Denote the angle by θπ when measured in the external domain D γ (Figure 2 right). Such contours correspond to maps Ψ : {| µ | &gt; 1 } → D γ such that with the standard branch of ( µ -1) θ and some constant c Ψ &gt; 0 . We will refer to such Ψ as corner maps and to the respective generalized SGD as corner algorithms . Formally,

<!-- formula-not-decoded -->

so we are interested in θ &gt; 1 . At the same time, we cannot take θ &gt; 2 , since this would violate the stability condition Ψ {| µ | &gt; 1 } ∩ spec( H ) = ∅ . Thus, the relevant range of values for θ is [1 , 2] . Within this range, increasing θ should have a positive α eff -related effect but a negative U Σ -related effect, since the contour γ = Ψ( | µ | = 1) is getting closer to the spectral segment [0 , λ max ] , thus amplifying the singularity | Ψ( e iϕ ) -λ k | -2 in Eq. (23). Our main technical result is

̸

Theorem 3 (C) . Let Ψ be a holomorphic function in { µ ∈ C : | µ | &gt; 1 } commuting with complex conjugation and obeying power law condition (24) with some 1 &lt; θ &lt; 2 . Assume that Ψ extends continuously to a C 1 function on the closed domain | µ | ≥ 1 , Ψ( µ ) →∞ as µ →∞ , and d dµ Ψ( µ ) = O ( | µ -1 | θ -1 ) as µ → 1 . Assume also that Ψ( { µ ∈ C : | µ | ≥ 1 , µ = 1 } ) ∩ [0 , λ max ] = ∅ , where λ max = λ 1 is the largest eigenvalue of H . Let power-law spectral assumptions (11) , (12) hold with some ν &gt; 1 , 0 &lt; ζ &lt; 2 . Then propagators (19) , (21) obey the following t →∞ asymptotics.

1. (Noise propagators) U t = C U t θ/ν -2 (1 + o (1)) , with the coefficient

<!-- formula-not-decoded -->

2. (Signal propagators) V t = C V t -θζ (1 + o (1)) , with the coefficient

<!-- formula-not-decoded -->

246

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

264

265

266

267

268

269

270

271

272

273

274

275

276

277

278

279

280

We see that the leading t → ∞ asymptotics of the propagators are completely determined by the λ ↘ 0 spectral asymptotics of the problem and the µ → 1 singularity of the map Ψ . The functions F U , F V can be written in terms of the Mittag-Leffler functions E θ,θ , E θ (see Section C).

Availability of the coefficients C U , C V ensures that the leading asymptotics of U t , V t are strict power laws with specific exponents 2 -θ/ν and θζ , respectively. Increasing θ indeed improves convergence of the signal propagators, but degrades convergence of the noise propagators.

The largest acceleration of the loss exponent ζ possibly achievable with corner algorithms is by a factor θ arbitrarily close to 2, but in general it will be lower since, by Theorem 1, the exponent of L t is the lower of the exponents of U t and V t ; accordingly, the optimal θ is obtained by balancing the two exponents, i.e. setting θζ = 2 -θ/ν . Also, we need the noise exponent 2 -θ/ν to be &gt; 1 , since otherwise the total noise coefficient U Σ = ∞ and L t diverges for any batch size | B | &lt; ∞ .

Combining these considerations, we get the phase diagram of feasible accelerations (Figure 1 right).

Theorem 4. Consider a problem with power-law spectral conditions (11) , (12) in the signaldominated phase, i.e. ν &gt; 1 , 0 &lt; ζ &lt; 2 -1 /ν . Let θ max denote the supremum of those θ for which there exists a corner algorithm and batch size B such that L t = O ( t -θζ ) . Then

<!-- formula-not-decoded -->

The phase diagram thus has three regions:

- I. Fully accelerated : θ max = 2 , achieved for ν &gt; 2 , 0 &lt; ζ &lt; 1 -1 /ν.
- III. Limited by U Σ -finiteness : θ max = ν &lt; 2 , 1 &lt; ν &lt; 2 , 0 &lt; ζ &lt; 1 /ν . The signal exponent θ max ζ is less than the noise exponent 2 -θ max /ν, but increasing θ makes U Σ diverge.
- II. Signal/noise balanced : θ max = 2 ζ +1 /ν &lt; 2 , max(1 /ν, 1 -1 /ν ) &lt; ζ &lt; 2 -1 /ν . The condition 1 /ν &lt; ζ ensures that U Σ is finite and less than 1 for | B | large enough.

## 5 Finite-memory approximations of corner algorithms

Though corner maps Ψ are irrational, they can be efficiently approximated by rational functions. It was originally famously discovered by [17] that the function | x | can by approximated by orderM rational functions with error O ( e -c √ M ) . This result was later refined in various ways. In particular, [12] establish a rational approximation with a similar error bound for general power functions z ↦→ z θ on complex domains. For θ ∈ (0 , 1) , this is done by writing

<!-- formula-not-decoded -->

and then approximating the last integral by the trapezoidal rule with uniform spacing h = π √ 2 θ/M .

In our setting, we start by explicitly defining a θ -corner map. This can be done in many ways; we find it convenient to set

<!-- formula-not-decoded -->

with a scaling parameter A &gt; 0 .

Proposition 1 (D) . For any 1 &lt; θ &lt; 2 , Eq. (28) defines a holomorphic map Ψ : C \ [0 , 1] → C such that where z θ denotes the standard branch in C \ ( -∞ , 0] . Also, Ψ( {| µ | ≥ 1 } ) ∩ (0 , 2 A ] = ∅ .

<!-- formula-not-decoded -->

Following [12], we approximate the last integral in Eq. (28) as

<!-- formula-not-decoded -->

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

Figure 3: Training loss and final predictions of the kernel model (220) trained to fit the target y ( x ) = 1 [ 1 / 4 , 3 / 4 ] ( x ) using either plain or corner SGD with batch size | B | = 100 . The loss trajectories oscillate strongly, so their smoothed versions are also shown and used to estimate the exponents ζ in power laws L t ∝ t -ζ . Corner SGD has θ = 1 . 8 and is approximated using finite memory M = 5 as in Proposition 2. We see that Corner SGD indeed accelerates the power-law convergence exponent of plain SGD. See Section F for details.

<!-- image -->

with some fixed constant l . Note that in contrast to (27), our integral and discretization are 'onesided' ( s &gt; 0 ), reflecting the fact that the corner map Ψ( µ ) is power law only at µ → 1 , which is related to the s → + ∞ behavior of the integrand.

Proposition 2 (E) . Let h = l/ √ M and

Let Ψ ( M ) denote the map Ψ discretized with M nodes by scheme (30). Observe that Ψ ( M ) is a rational function, Ψ ( M ) = P Q , where deg P = M + 1 and deg Q ≤ M (in particular, P ( µ ) = ( µ -1) ∏ M m =1 ( µ -1 + e -( m -1 / 2) h ) ). We can then associate to Ψ ( M ) a memoryM algorithm (1) with particular α, b , c , D , for example as follows.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then the respective characteristic polynomial χ ( µ ) = P ( µ ) -λQ ( µ ) with P Q = Ψ ( M ) .

<!-- formula-not-decoded -->

Of course, as any stationary finite-memory algorithm, for very large t the M -discretized corner algorithm can only provide a O ( t -ζ ) convergence of the loss. But, thanks to the O ( e -c √ M ) rational approximation bound, we expect that even with moderate M , for practically relevant finite ranges of t the convergence should be close to O ( t -θζ ) of the ideal corner algorithm.

Experiments with a synthetic problem and MNIST confirm that corner algorithms accelerate the exponents of plain SGD (see Appendix F and Figure 3). We also provide additional discussion of corner algorithms in Appendix G. In particular, we note that, while corner algorithms require significantly more memory than plain SGD, the amount of computation they perform is typically not much larger than for SGD. Our theoretical results significantly depended on the SE assumption (6) with τ 2 = 0 , but it appears that the theory can be extended to a more general setting (at the cost of more complicated expansions).

## References

- [1] Alexander Atanasov, Blake Bordelon, and Cengiz Pehlevan. Neural networks as kernel learners: The silent alignment effect. arXiv preprint arXiv:2111.00034 , 2021.

- [2] Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. Explaining 304 neural scaling laws. arXiv preprint arXiv:2102.06701 , 2021. 305
- [3] Ronen Basri, Meirav Galun, Amnon Geifman, David Jacobs, Yoni Kasten, and Shira Kritch306 man. Frequency bias in neural networks for input of non-uniform density. In International 307 Conference on Machine Learning , pages 685-694. PMLR, 2020. 308
- [4] Raphaël Berthier, Francis Bach, and Pierre Gaillard. Tight nonparametric convergence rates for 309 stochastic gradient descent under the noiseless linear model. arXiv preprint arXiv:2006.08212 , 310 2020. 311
- [5] MŠBirman and M Z Solomjak. Asymptotic behavior of the spectrum of weakly polar integral 312 operators. Mathematics of the USSR-Izvestiya , 4(5):1151-1168, oct 1970. 313
- [6] Blake Bordelon, Alexander Atanasov, and Cengiz Pehlevan. How feature learning can improve 314 neural scaling laws. arXiv preprint arXiv:2409.17858 , 2024. 315
- [7] Blake Bordelon and Cengiz Pehlevan. Learning curves for sgd on structured features. arXiv 316 preprint arXiv:2106.02713 , 2021. 317
- [8] Helmut Brakhage. On ill-posed problems and the method of conjugate gradients. In Inverse 318 and ill-posed Problems , pages 165-175. Elsevier, 1987. 319
- [9] Andrea Caponnetto and Ernesto De Vito. Optimal rates for the regularized least-squares algo320 rithm. Foundations of Computational Mathematics , 7(3):331-368, 2007. 321
- [10] Hugo Cui, Bruno Loureiro, Florent Krzakala, and Lenka Zdeborová. Generalization error rates 322 in kernel regression: The crossover from the noiseless to noisy regime. Advances in Neural 323 Information Processing Systems , 34, 2021. 324
- [11] J.B. Garnett and D.E. Marshall. Harmonic Measure . New Mathematical Monographs. Cam325 bridge University Press, 2005. 326
- [12] Abinand Gopal and Lloyd N Trefethen. Representation of conformal maps by rational func327 tions. Numerische Mathematik , 142:359-382, 2019. 328
- [13] Hans J Haubold, Arak M Mathai, and Ram K Saxena. Mittag-leffler functions and their appli329 cations. Journal of applied mathematics , 2011(1):298628, 2011. 330
- [14] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and 331 generalization in neural networks. arXiv preprint arXiv:1806.07572 , 2018. 332
- [15] Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. ATT Labs 333 [Online]. Available: http://yann.lecun.com/exdb/mnist , 2, 2010. 334
- [16] Arkadi S Nemirovskiy and Boris T Polyak. Iterative methods for solving linear ill-posed 335 problems under precise information. Eng. Cyber. , (4):50-56, 1984. 336
- [17] Donald J Newman. Rational approximation to | x | . Michigan Mathematical Journal , 11(1):11, 337 1964. 338
- [18] Elliot Paquette, Courtney Paquette, Lechao Xiao, and Jeffrey Pennington. 4+ 3 phases of 339 compute-optimal neural scaling laws. arXiv preprint arXiv:2405.15074 , 2024. 340
- [19] Boris T Polyak. Some methods of speeding up the convergence of iteration methods. Ussr 341 computational mathematics and mathematical physics , 4(5):1-17, 1964. 342
- [20] Sumit Roy and John J Shynk. Analysis of the momentum lms algorithm. IEEE transactions 343 on acoustics, speech, and signal processing , 38(12):2088-2098, 1990. 344
- [21] P Stiller. An introduction to the theory of resultants. Mathematics and Computer Science, 345 T&amp;M University, Texas, College Station, TX , 1996. 346
- [22] Mehmet Ali Tugay and Yalcin Tanik. Properties of the momentum lms algorithm. Signal 347 Processing , 18(2):117-127, 1989. 348
- [23] Aditya Varre and Nicolas Flammarion. Accelerated sgd for non-strongly-convex least squares. 349 arXiv preprint arXiv:2203.01744 , 2022. 350
- [24] Aditya Varre, Loucas Pillaud-Vivien, and Nicolas Flammarion. Last iterate convergence of sgd 351 for least-squares in the interpolation regime. arXiv preprint arXiv:2102.03183 , 2021. 352

- [25] Maksim Velikanov, Denis Kuznedelev, and Dmitry Yarotsky. A view of mini-batch sgd via 353 generating functions: conditions of convergence, phase transitions, benefit from negative mo354 menta. In The Eleventh International Conference on Learning Representations (ICLR 2023) , 355 2023. 356
- [26] Maksim Velikanov and Dmitry Yarotsky. Explicit loss asymptotics in the gradient descent 357 training of neural networks. Advances in Neural Information Processing Systems , 34, 2021. 358
- [27] Greg Yang and Hadi Salman. A fine-grained spectral perspective on neural networks, 2020. 359
- [28] Dmitry Yarotsky. Collective evolution of weights in wide neural networks. arXiv preprint 360 arXiv:1810.03974 , 2018. 361
- [29] Dmitry Yarotsky and Maksim Velikanov. Sgd with memory: fundamental properties and 362 stochastic acceleration. arXiv preprint arXiv:2410.04228 , 2024. 363

## Contents 364

|   365 | 1          | Introduction                                            |   1 |
|-------|------------|---------------------------------------------------------|-----|
|   366 | 2          | Background                                              |   2 |
|   367 | 3          | The contour view of generalized (S)GD                   |   5 |
|   368 | 4          | Corner algorithms                                       |   7 |
|   369 | 5          | Finite-memory approximations of corner algorithms       |   8 |
|   370 | References | References                                              |   9 |
|   371 | A          | Derivations of Section 3                                |  12 |
|   372 | B          | Memory-1 contours                                       |  12 |
|   373 | C          | Proof of Theorem 3                                      |  16 |
|   374 |            | C.1 The noise propagators . . . . . . . . . . . . . . . |  16 |
|   375 |            | C.2 The signal propagators . . . . . . . . . . . . . .  |  21 |
|   376 | D          | Proof of Proposition 1                                  |  26 |
|   377 | E          | Proof of Proposition 2                                  |  27 |
|   378 | F          | Experiments                                             |  28 |
|   379 | G          | Additional notes and discussion                         |  29 |
|   380 | H          | The synthetic 1D example                                |  31 |
|   381 | I          | Extending the proof of Theorem 3 to τ 2 = 0             |  32 |

̸

387

388

389

390

391

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, by Sherman-Morrison formula and the above identity, 385

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using ( 1 0 T )( µ -S 0 ) -1 ( 1 0 ) = 1 µ -1 , it follows that 386

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B Memory-1 contours

In figure 4 we show different contours γ = Ψ( {| µ | = 1 } ) corresponding to memory-1 algorithms (see Section 3 for the introduction of contours). Below we discuss memory-1 algorithms and their contours in the order of increasing generality.

Plain (S)GD. In (S)GD with learning rate α &gt; 0 we have P ( µ ) = µ -1 and Q ( µ ) = -α, so

Thus, γ is the circle | z -1 α | = 1 α . 392

<!-- formula-not-decoded -->

Figure 4: Contours γ = Ψ( { µ : | µ | = 1 } ) corresponding to different memory-1 maps Ψ (see Section B). Left: plain Gradient Descent (a circle). Center: Heavy Ball (an ellipse; β = 0 . 5 ). Right: general memory-1 algorithms (a Zhukovsky airfoil; β = 0 . 65 , q 0 = 0 . 125 , q 1 = -1 ).

<!-- image -->

## A Derivations of Section 3 382

We have 383

It follows that 384

Heavy Ball. Heavy Ball with learning rate α and momentum parameter β has standard stability 393 conditions α &gt; 0 , β ∈ ( -1 , 1) and λ max &lt; 2+2 β α [20, 22]. We have P ( µ ) = ( µ -1)( µ -β ) and 394 Q ( µ ) = -αµ, so 395

If | µ | = 1 , then µµ = 1 and hence 396

<!-- formula-not-decoded -->

Writing µ = x + iy, we get 397

<!-- formula-not-decoded -->

It follows that γ is an ellipse with the semi-axis 1+ β α along x and the semi-axis 1 -β α along y . The 398 learning rate α determines the size of the ellipse while the momentum parameter β determines its 399 shape. If β &gt; 0 , then the ellipse is elongated in the x direction, and otherwise in the y direction. 400 Assuming β &gt; 0 , the eccentricity of the ellipse equals e = √ 1 -(1 -β ) 2 / (1 + β ) 2 = 2 √ β 1+ β . Plain 401 GD is the special case of Heavy Ball with β = 0 . 402

403

404

405

406

407

General memory-1 (S)GD. In a general memory-1 algorithm we have P ( µ ) = ( µ -1)( µ -β ) and Q ( µ ) = q 0 + q 1 µ , so

<!-- formula-not-decoded -->

Heavy Ball is the special case of general memory-1 algorithms with q 0 = 0 .

In [29] it was shown that on the spectral interval (0 , λ max ] the strict stability of the generalized memory-1 SGD is equivalent to the conditions

<!-- formula-not-decoded -->

(note that the Heavy Ball stability conditions result by setting q 0 = 0 , q 1 = -α ). 408

Zhukovsky airfoil representation. The map Ψ can be written as a composition of linear transfor409 mations and the Zhukovsky function 410

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Indeed, let 411

then 412

where 413

and √ r is imaginary if r &lt; 0 . 414

Thus, the contour γ = Ψ( {| µ | = 1 } ) is a rescaled image of a circle under the Zhukovsky transform, 415 i.e. a 'Zhukovsky airfoil'. 416

<!-- formula-not-decoded -->

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

̸

Conditions of injectivity. As discussed in Section 3, the case of maps Ψ injective on the domain | µ | &gt; 1 seems especially natural and attractive. Let us examine when the map Ψ given by Eq. (50) is injective. We can assume without loss that q 1 = 0 since otherwise the map Ψ is not locally injective at ∞ .

The Zhukovsky transform can be written as a composition of two linear fractional transformations and the function w = z 2 :

<!-- formula-not-decoded -->

The image of a generalized disc on the extended complex plane under a linear fractional map is again a generalized disc, and the map w = z 2 is injective on a generalized open disc if and only if the disc does not contain 0 and ∞ . Hence, a necessary and sufficient condition for J to be injective on a generalized open disc is that this disc not contain the points ± 1 . It follows that Ψ is injective on the generalized disc | µ | &gt; 1 iff

<!-- formula-not-decoded -->

Let us henceforth assume the stability condition -1 &lt; β &lt; 1 as given in Eq. (51). Consider separately the cases of negative and positive r .

1. r ≤ 0 corresponds to -1 ≤ q 0 q 1 ≤ -β . In this case condition (59) is equivalent to -1 ≤ q 0 q 1 , i.e. it holds.

However, the special case q 0 q 1 = -1 is the degenerate scenario in which the denominator of Ψ vanishes at µ = 1 and the stability condition q 1 &lt; -q 0 in Eq. (51) is violated, so we will discard this special case.

2. r &gt; 0 corresponds to q 0 q 1 &lt; -1 or q 0 q 1 &gt; -β . The option q 0 q 1 &lt; -1 is inconsistent with condition (59), leaving only the option q 0 q 1 &gt; -β .

<!-- formula-not-decoded -->

- (a) If q 0 q 1 ≤ 0 , then condition (59) is equivalent to

which holds true thanks to the assumption β &lt; 1 .

- (b) If q 0 q 1 ≥ 0 , then condition (59) is equivalent to

which holds iff

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Summarizing, assuming the stability condition -1 &lt; β &lt; 1 and excluding the degenerate case q 0 = -q 1 , the condition of injectivity of the map Ψ on the domain | µ | &gt; 1 reads

<!-- formula-not-decoded -->

We remark that this condition can also be reached in a different way. There are two obvious nec443 essary conditions of injectivity of Ψ on the set | µ | &gt; 1 : the absence of poles of Ψ and zeros of the 444 derivative Ψ ′ from this domain (the latter ensures the local injectivity). The absence of poles means 445 that -1 ≤ q 0 q 1 ≤ 1 . The zeros of the derivative are given by the equation 446

<!-- formula-not-decoded -->

Both roots of a quadratic equation µ 2 + aµ + b = 0 lie inside the closed unit circle iff | a | ≤ 1+ b ≤ 2 . 447 Applying this condition (and discarding the case q 0 /q 1 = -1 ), we reach the same inequalities (63). 448 In particular, the conditions of absence of poles and the roots of the derivative turn out to be not only 449 necessary, but also sufficient. 450

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

Algebraic equation of the contour. The circle | µ | = 1 is a real algebraic curve defined by the polynomial equation x 2 + y 2 = 1 , where µ = x + iy . Images of real algebraic curves under rational complex maps are again algebraic curves, and the corresponding equations can be found using, e.g., Macaulay resultants [21]. In the particular case of unit circle the computation can be performed in terms of standard resultants as follows.

Recall that Ψ( µ ) = P ( µ ) Q ( µ ) , where P is a polynomial of degree M + 1 , and Q is a polynomial of degree ≤ M ; we assume P and Q to have real coefficients. Denote w = Ψ( µ ) , then

<!-- formula-not-decoded -->

Since µ belongs to the unit circle, µµ = 1 . Applying complex conjugation and the identity µ = 1 /µ to the above equation, we get the second equation

<!-- formula-not-decoded -->

Note that ˜ Q ( µ ) = µ M +1 Q (1 /µ ) and ˜ P ( µ ) = µ M +1 P (1 /µ ) are polynomials in µ of degree M +1 or less. It follows that µ satisfies two polynomial conditions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

i.e. µ is a common root of two polynomials, T 1 ( µ ) and T 2 ( µ ) . Two polynomials have a common 463 root iff their resultant vanishes. The polynomials T 1 ( µ ) , T 2 ( µ ) have degree M + 1 or less and 464 include w and w linearly in their coefficients. It follows that the set Ψ( {| µ | = 1 } ) can be described 465 by the equation 466

<!-- formula-not-decoded -->

which is a polynomial equation in w and w of degree at most 2( M +1) . 467

We implement now this general program for M = 1 . Given quadratic polynomials 468

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where their resultant can be written as 469

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Considering real β, q 0 , q 1 and w = x + iy , we get 471

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It follows that the contour Ψ( {| µ | = 1 } ) can be described by the quartic (in general) equation 472

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As expected, in the Heavy Ball case q 0 = 0 this equation degenerates into the quadratic equation 473

<!-- formula-not-decoded -->

In our case 470

## C Proof of Theorem 3 474

## C.1 The noise propagators 475

The function F U . Let us introduce the values 476

<!-- formula-not-decoded -->

so that, by Eq. (19), the propagator U t can be written as 477

<!-- formula-not-decoded -->

With the change of variables ϕ = sλ 1 /θ , 478

<!-- formula-not-decoded -->

where we have denoted 479

483

484

485

486

487

488

489

490

491

492

493

494

495

496

<!-- formula-not-decoded -->

Recall that we assume Ψ( µ ) = -c Ψ ( µ -1) θ (1 + o (1)) as µ → 1 . By formally taking the limit 480 λ ↘ 0 in the integral, we then expect F U ( r, λ ) to converge to 481

<!-- formula-not-decoded -->

for any fixed r . This integral can be equivalently written as 482

<!-- formula-not-decoded -->

assuming the standard branch of z θ holomorphic in C \ ( -∞ , 0] .

The function F U can be viewed (up to a coefficient) as the inverse Fourier transform of the function s ↦→ ( c Ψ e i (sign s ) θπ/ 2 | s | θ + 1) -1 . Note that, thanks to the condition θ &gt; 1 , the latter function is Lebesgue-integrable, so F U ( r ) is well-defined and continuous for all r ∈ R . The function F U can also be written in terms of the special Mittag-Leffler function E θ,θ (see its integral representation (6.8) in [13]):

<!-- formula-not-decoded -->

The following asymptotic properties of F U ( r ) can be derived from the general asymptotic expansions of Mittag-Leffler functions (sections 1 and 6 in [13]), but we provide proofs for completeness.

where the integration path γ encircles the cut ( -∞ , 0] and the singularities of the denominator.

## Lemma 1.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. 1. Consider the function f ( z ) integrated in Eq. (86). For any r ∈ R and θ ∈ (1 , 2) , 497 the function f is holomorphic in any strip T a = { 0 &lt; ℜ z &lt; a } , a &gt; 0 , and is bounded in T a 498 as | f ( z ) | = O ( | z | -θ ) . It follows that the integration line i R can be deformed to i R + a without 499 changing the integral. If r &lt; 0 , then by letting a → + ∞ we can make the integral arbitrarily small. 500

2. By the change of variables rz = z ′ , 501

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where 502

We can find lim r ↘ 0 u ( r ) as follows. Observe that the integration line i R can be deformed to the 503 line γ a , a &gt; 0 , encircling the negative semi-axis: 504

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Indeed, if r is sufficiently small, then this deformation occurs within the holomorphy domain of the 505 integrated function. The integral is preserved since θ &gt; 0 and since we deform in the half-plane 506 where the argument of e z ′ has ℜ z ′ &lt; 0 . 507

Thus, for any fixed a &gt; 0 we have 508

<!-- formula-not-decoded -->

where in the last step we integrated by parts. In the last integral, thanks to the weakness of the 509 singularity z ′ 1 -θ at z ′ = 0 (note that 1 -θ &gt; -1 ), we can let a → 0 : 510

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the last step we used the identity Γ( z )Γ(1 -z ) = π sin( πz ) . This is essentially Hankel's 511 representation of the Gamma function, valid for all θ ∈ C by analytic continuation. Summarizing, 512

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

3. We start by performing integration by parts in F U : 513

<!-- formula-not-decoded -->

Performing again the change of variables rz = z ′ , we have 514

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where 515

516

517

518

with some constant c 1 &gt; 0 and c 0 .

Note that the integrated function has two singular points z ′ ∈ C \ ( -∞ , 0] where the denominator 519 c Ψ ( z ′ /r ) θ +1 = 0 . These two points depend linearly on r . Require additionally that γ ′ lie to the 520 right of these points for all r &gt; 0 , so that i R can be deformed to γ ′ without meeting the singularities. 521

To compute lim r →∞ v ( r ) , we again transform the integration line. Let γ ′ be a line that lies in the domain C \ ( -∞ , 0) and can be represented as the graph of a function ℜ z = f ( ℑ z ) such that

<!-- formula-not-decoded -->

This requirement is feasible with a small enough c 1 &gt; 0 since, by the condition θ &lt; 2 , the imaginary 522 parts of the singular points are negative. 523

524

525

526

With these assumptions, integration in Eq. (101) can be changed to integration over γ ′ . Thanks to condition (102), the integrand converges exponentially fast at z ′ → ∞ , and we can take the limit r → + ∞ :

527

528

529

<!-- formula-not-decoded -->

The contour γ ′ can now be transformed to a contour encircling the negative semi-axis, and applying Eq. (97) we get

<!-- formula-not-decoded -->

The formal leading term in U t . We have 530

<!-- formula-not-decoded -->

To extract the leading term in this expression, we set the second argument in F U ( tλ 1 /θ k , λ k ) to 0: 531

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where 532

533

Lemma 2.

<!-- formula-not-decoded -->

Proof. Note first that the integral on the right is convergent. Indeed, by statement 2 of Lemma 1, r 1 -θ/ν F 2 ( r ) ∝ r 1 -θ/ν +2( θ -1) = r θ (2 -1 /ν ) -1 near r = 0 . Since we assume ν &gt; 1 and θ &gt; 1 , the function r 1 -θ/ν F 2 ( r ) is bounded near r = 0 . Also, by statement 3 of Lemma 1, r 1 -θ/ν F 2 ( r ) ∝ r 1 -θ/ν -2( θ +1) = O ( r -3 ) as r → + ∞ .

<!-- formula-not-decoded -->

For any interval I in R + , denote by S I,t the part of the expansion (107) of a t corresponding to the terms with tλ 1 /θ k ∈ I :

Recall that the eigenvalues λ are ordered and λ k = Λ k -ν (1 + o (1)) by capacity condition (11). It follows that for a given fixed number r &gt; 0 , the condition tλ 1 /θ k &gt; r holds whenever k &lt; k r , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

534

535

536

537

538

539

540

541

Then, for I = [ u, v ] with 0 &lt; u &lt; v &lt; ∞ we have 542

<!-- formula-not-decoded -->

Moreover, for any interval I = [ u, v ] with 0 &lt; u &lt; v &lt; ∞ we can approximate ∫ I r 2 F 2 U ( r ) dr -θ/ν 543 by integral sums corresponding to sub-divisions I = I 1 ∪ I 2 ∪ . . . ∪ I n , apply the above inequalities 544 to each I s , and conclude that 545

<!-- formula-not-decoded -->

It remains to handle the two parts of a t corresponding to the remaining intervals I = [0 , u ] and 546 I = [ v, ∞ ) . It suffices to show that the associated contributions S I,t can be made arbitrarily small 547 uniformly in t by making u small and v large enough. 548

549

550

551

Consider first the interval I = [ v, ∞ ) . Note that by Lemma 1 for all r &gt; 1 we can write

<!-- formula-not-decoded -->

with some constant C , and we also have for all k

<!-- formula-not-decoded -->

for suitable constants Λ -, Λ + . It follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with O (1) denoting an expression bounded by a t, v -independent constant. This is the desired con552 vergence property of S I,t . 553

Similarly, for the other interval I = [0 , u ] we use the inequality 554

<!-- formula-not-decoded -->

555

556

557

558

559

560

561

562

563

564

also following by Lemma 1. Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is the desired convergence property of since

S I,t ν &gt; 1 .

Completion of proof. Wehave shown that if we replace F U ( tλ 1 /θ k , λ k ) by F U ( tλ 1 /θ k ) in Eq. (105), we get desired asymptotics of U t in the limit t → + ∞ . We will show now that this replacement introduces a lower-order correction o ( t θ/ν -2 ); this will complete the proof.

We start with a technical lemma (to be applied with f = Ψ ) giving a lower bound for deviations of asymptotic power law functions with θ &lt; 2 from real values.

̸

Lemma3. Suppose that f : { µ ∈ C : | µ | = 1 } → C is continuous, f ( µ ) = -c ( µ -1) θ (1+ o (1)) as µ → 1 with some θ ∈ [0 , 2) and c &gt; 0 . Suppose also that f ( { µ ∈ C : | µ | = 1 , µ = 1 } ) ∩ [0 , λ max ] = ∅ for some λ max &gt; 0 . Then there exist a constant C &gt; 0 such that

<!-- formula-not-decoded -->

̸

Proof. If we fix any small ϵ &gt; 0 , then, by the condition f ( { µ ∈ C : | µ | = 1 , µ = 1 } ) ∩ [0 , λ max ] = 565 ∅ and a compactness argument, there exist C ′ , C &gt; 0 such that 566

<!-- formula-not-decoded -->

It remains to establish inequality (125) for | s | &lt; ϵ . Since f ( µ ) = c ( µ -1) θ (1+ o (1)) and θ ∈ [0 , 2) , 567

<!-- formula-not-decoded -->

for | s | small enough. 568

569

570

571

572

573

574

Lemma 4.

1. | F U ( r, λ ) -F U ( r ) | = o (1) as λ → 0 , uniformly in all r ∈ R .

2. F U ( r, λ ) = O ( 1 r ) for all r of the form r = tλ 1 /θ , t = 1 , 2 , . . . , uniformly in all λ ∈ (0 , λ max ] .

Proof. 1. It suffices to show that, as λ ↘ 0 , the functions

<!-- formula-not-decoded -->

converge in L 1 ( R ) to

580

581

582

583

584

<!-- formula-not-decoded -->

Let us divide the interval [ -π/λ 1 /θ , π/λ 1 /θ ] into two subsets: 575

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where h is some fixed number such that 1 θ 2 &lt; h &lt; 1 θ . 576

By Lemma 3, | Ψ( e isλ 1 /θ ) /λ -1 | ≥ c | s | θ uniformly for all s ∈ [ -π/λ 1 /θ , π/λ 1 /θ ] and λ ∈ 577 (0 , λ max ] . It follows that 578

<!-- formula-not-decoded -->

for some constant c &gt; 0 . Using the condition 1 θ 2 &lt; h , it follows that 579

<!-- formula-not-decoded -->

Thus, we can assume without loss that the functions f λ vanish outside the intervals I 1 ( λ ) . On these intervals, thanks to the condition h &lt; 1 θ , we have

<!-- formula-not-decoded -->

uniformly in s ∈ I 1 ( λ ) . We can then apply the dominated convergence theorem to the functions | f λ -f 0 | , with a dominating function C (1 + | s | θ ) -1 , and conclude that f λ → f 0 in L 1 ( R ) , as desired.

2. We start by performing integration by parts in U ( t, λ ) : 585

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

implying 586

We will show that this integral is O ( 1 λ ) . 587

Note first that we can replace the integration on [ -π, π ] by integration on [ -a, a ] for any 0 &lt; 588 a &lt; π . Indeed, by our assumptions Ψ is C 1 on the unit circle, and Ψ( µ ) = 1 there only if µ = 1 . 589 Accordingly, the remaining part of the integral is non-singular as λ ↘ 0 and so is uniformly bounded 590 for all λ ∈ (0 , λ max ] . 591

<!-- formula-not-decoded -->

Recall that by our assumption Ψ ′ ( µ ) = O ( | µ -1 | θ -1 ) as µ → 1 . Applying again Lemma 3, 592

with some constant C ′ independent of t, λ . It follows that 593

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as claimed. 594

We return now to proving that replacing F U ( tλ 1 /θ k , λ k ) by F U ( tλ 1 /θ k ) in Eq. (105) amounts to a 595 lower-order correction o ( t θ/ν -2 ) . It suffices to prove that ∆ a t → 0 , where 596

<!-- formula-not-decoded -->

For any interval I ⊂ R , denote by ∆ S I,t the part of ∆ a t corresponding to the terms in (144) such 597 that tλ 1 /θ k ∈ I . By statement 1 of Lemma 4, for any u &gt; 0 we have, as t →∞ , 598

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used the fact that 2 ν/θ &gt; ν &gt; 1 . 599

Now consider the remaining interval I = [ u, + ∞ ) . It suffices to prove that | ∆ S [ u, + ∞ ) ,t | can be 600 made arbitrarily small uniformly in t by choosing u large enough. By statement 2 of Lemma 4, we 601 can write 602

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with some t, u -independent constant C ′ . This completes the proof of statement 1 of Theorem 3. 603

604

605

606

## C.2 The signal propagators

The proof for the signal propagators follows the same ideas as for the noise propagators, with appropriate adjustments.

The function F V . We introduce the values 607

<!-- formula-not-decoded -->

so that, by Eq. (21), the propagators V t can be written as 608

<!-- formula-not-decoded -->

With the change of variables ϕ = sλ 1 /θ , 609

<!-- formula-not-decoded -->

where 610

We again recall that Ψ( µ ) = -c Ψ ( µ -1) θ (1+ o (1)) as µ → 1 and formally take the pointwise limit 611 λ ↘ 0 in the integrand to obtain the expression 612

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any fixed r . This integral can be equivalently written as 613

<!-- formula-not-decoded -->

assuming again the standard branch of z θ holomorphic in C \ ( -∞ , 0] . The function F V can be 614 written in terms of the Mittag-Leffler function E θ ≡ E θ, 1 (the special case of E a,b given by Eq. 615 (87)): 616

<!-- formula-not-decoded -->

Note that, in contrast to F U , the integrals (156), (157) are not absolutely summable, due to the z -1 fall off of the integrand at z → ∞ . However, the integrand is square-summable and so F V , as a Fourier transform of such function, is well-defined almost everywhere as a square-integrable function.

̸

In fact, F V can be defined for each particular r = 0 by restricting the integration in (156) to segments [ u, v ] and letting u → -∞ and v → ∞ . Indeed, the resulting Fourier transforms F ( u,v ) V converge to F V in L 2 ( R ) . However, these transforms are continuous functions of r , and as u →∞ , v →∞ they converge pointwise, and even uniformly on the sets { r : | r | &gt; ϵ } , for any fixed ϵ &gt; 0 .

To see this last property of uniform pointwise convergence, note that the integrand in (156) has the form ( s -1 + O ( s -1 -θ )) e irs as s →∞ . The component O ( s -1 -θ )) is in L 1 , so the respective part of F ( u,v ) V converges as u →-∞ , v →∞ uniformly for all r ∈ R . Regarding the s -1 component, integrating by parts gives

<!-- formula-not-decoded -->

This expression converges as v →∞ uniformly for { r : | r | &gt; ϵ } with any fixed ϵ &gt; 0 , as claimed. The same argument applies to ∫ -1 u .

The above argument shows, in particular, that F V is naturally defined as a function continuous on the intervals (0 , + ∞ ) and ( -∞ , 0) .

We collect further properties of F V ( r ) in the following lemma that parallels Lemma 1 for F U . The proofs are also similar to the proofs in Lemma 1.

## Lemma 5.

1. F V ( r ) = 0 for r &lt; 0 .
2. F V ( r ) → 1 as r ↘ 0 .

<!-- formula-not-decoded -->

617

618

619

620

621

622

623

624

625

626

627

628

629

630

631

632

633

634

635

636

637

638

Proof. 1. Like in Lemma 1, this follows by deforming the integration line in Eq. (157) towards 639 + ∞ . 640

2. By the change of variables rz = z ′ , 641

<!-- formula-not-decoded -->

As in Lemma 1, the integration line i R can be deformed to the line γ a , a &gt; 0 , encircling the negative 642 semi-axis: 643

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking the limit r ↘ 0 , we get 644

since the last integral simply amounts to the residue of e z ′ /z ′ at z ′ = 0 . 645

3. Using the same contour γ ′ as in Lemma 1, 646

<!-- formula-not-decoded -->

Taking the limit r → + ∞ and deforming the contour to the negative semi-axis as in Lemma 1, 647

648

The formal leading term in V t . We have 649

<!-- formula-not-decoded -->

To extract the leading term in this expression, we set the second argument in F V ( tλ 1 /θ k , λ k ) to 0: 650

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where 651

The analog of Lemma 2 is 652

Lemma 6.

<!-- formula-not-decoded -->

Proof. First, observe that, by the source condition (12) and Lemma 5, the integral converges near 653 r = 0 since θζ &gt; 0 , and near r = ∞ since ζ &lt; 2 . 654

We can establish convergence of the sequence b t using the same steps as in Lemma 2. We first 655 introduce the sums S I,t comprising the terms of expansion (170) such that tλ 1 /θ k ∈ I . For intervals 656 I = [ u, v ] with 0 &lt; u &lt; v &lt; ∞ we show, using the source condition (12) and approximation by 657 integral sums, that 658

<!-- formula-not-decoded -->

After that we show that the contribution of the remaining intervals ( v, + ∞ ) and (0 , u ) can be made 659 arbitrarily small uniformly in t by adjusting u, v . 660

In particular, consider the interval I = ( v, + ∞ ) . Let R ( λ ) = ∑ k : λ k ≤ λ λ k ( e T k w ∗ ) 2 denote the 661 cumulative distribution function of the spectral measure. Since the spectral measure is compactly 662 supported, assumption (12) implies that R ( λ ) ≤ Q ′ λ ζ for all λ &gt; 0 with some Q ′ &gt; 0 . Using 663 statement 3 of Lemma 5 and integration by parts, we can bound 664

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with some constant C ′ independent of v, t . 665

For the intervals I = (0 , u ) we have 666

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Completion of proof. It remains to show that the correction in V t due to the replacement of F V ( tλ 1 /θ k , λ k ) by F V ( tλ 1 /θ k ) in Eq. (168) is o ( t -θζ ) . We first establish an analog of Lemma 4:

Lemma 7. Assuming that r = tλ 1 /θ with t = 1 , 2 , . . . :

<!-- formula-not-decoded -->

2. | F V ( r, λ ) | ≤ C min( 1 r , 1) for all t = 1 , 2 , . . . and λ ∈ (0 , λ max ] , with some r, λ -independent constant C .

Proof. 1. The proof of this property is more complicated than the earlier proof for F U because the integrals defining F V are not absolutely convergent. Recall the integration by parts argument (159) used to define F V ( r ) as the pointwise limit of the functions F ( u,v ) V ( r ) . We extend this approach to the functions F V ( r, λ ) with λ &gt; 0 . Specifically, let F ( u ) V ( r, λ ) be defined as F V ( r, λ ) in Eq. (154), but with integration restricted to the segment [ -u, u ] . By analogy with our convention F V ( r ) ≡ F V ( r, λ = 0) , denote also F ( u ) V ( r ) ≡ F ( u ) V ( r, λ = 0) . We will establish the following two properties:

(a)

|

F

(

u

V

(

r, λ

)

-

F

V

(

r, λ

)

| ≤

C

ru for all

0

&lt; λ &lt; λ

max with a

r, u, λ

-independent constant

C

.

<!-- formula-not-decoded -->

Observe first that these two properties imply the claimed uniform convergence | F V ( r, λ ) -F V ( r ) | = o (1) as λ → 0 . Indeed, given any δ &gt; 0 , first set u = 3 C ϵ so that by (a) we have

<!-- formula-not-decoded -->

)

667

668

669

670

671

672

673

674

675

676

677

678

679

680

681

682

683

684

685

for all r &gt; ϵ and 0 &lt; λ &lt; λ max . This inequality also holds in the limit λ ↘ 0 , i.e. 686

687

688

689

690

691

To prove statement (a), we perform integration by parts, using the 2 π λ 1 /θ -periodicity of the integrand: 692

<!-- formula-not-decoded -->

Now (b) implies that for sufficiently small λ we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

uniformly in r ∈ R . Combining all three above inequalities, we see that for sufficiently small λ

uniformly for r &gt; ϵ, as desired.

It remains to prove the statements (a) and (b). Statement (b) immediately follows from the uniform λ ↘ 0 convergence of the integrand in expression (154) on the interval s ∈ [ -u, u ] .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By our assumptions on Ψ , Lemma 3 and standard inequalities, there exist λ, s -independent constants 693 C, c &gt; 0 such that for all λ ∈ (0 , λ max ] and s ∈ [ -π λ 1 /θ , π λ 1 /θ ] 694

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Applying these inequalities to Eq. (187), we find that 695

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as desired. 696

697

698

699

700

701

702

2. Note that simply by setting u = 0 in the bound (192), since the first term on the r.h.s. of (192) vanishes and the second converges thanks to θ &gt; 1 .

It remains to prove that F V ( r, λ ) is bounded uniformly in r, λ . It suffices to prove this for r &lt; ϵ with some fixed ϵ &gt; 0 , since for larger r this follows from bound (194). Since r = tλ 1 /θ , this means it is sufficient to consider

<!-- formula-not-decoded -->

To this end consider the original definition (151) of V ( t, λ ) in terms of integration over the contour 703 {| µ | = 1 } . We will deform this contour within the analiticity domain { µ ∈ C : | µ | ≥ 1 } to another 704 contour γ , to be specified below, that fully encircles the point µ = 1 : 705

<!-- formula-not-decoded -->

It is convenient to subtract the residue of µ t -1 / ( µ -1) equal to 1: 706

<!-- formula-not-decoded -->

We define now γ as the original contour perturbed to include an arc of radius 1 /t centered at 1: 707

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ϕ 1 ∈ (0 , π 2 ) , ϕ 2 ∈ ( π 2 , π ) are such that γ is connected. Note that ϕ 1 ∝ 1 t as t →∞ .

708

Now we bound separately the contribution to the integral from γ 1 and γ 2 . For γ 1 and -π ≤ ϕ ≤ π 709 we use the inequalities 710

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with a ϕ, λ -independent constant c &gt; 0 . This gives, using Eq. (195), 711

<!-- formula-not-decoded -->

For the γ 2 component we use the inequalities 712

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(Inequality (205) relies on the assumption θ &lt; 2 and can be proved similarly to Lemma 3.) This 713 gives 714

<!-- formula-not-decoded -->

715

716

Fixing some ϵ &gt; 0 , we see from Eqs. (203), (206) that under assumption (195) the expressions | V ( t, λ ) -1 | , and hence | V ( t, λ ) | , are uniformly bounded, as desired.

717

718

719

720

721

722

723

724

725

726

727

728

This completes the proof of the lemma.

This lemma can now be used to show that replacing F V ( tλ 1 /θ k , λ k ) by F V ( tλ 1 /θ k ) in Eq. (168) amounts to a lower-order correction o ( t -θζ ) in the propagator V t . The argument is similar to the respective argument for F U in the end of Section C.1. Statement 1 of Lemma 7 is used to show this for the contribution of the terms k with u &lt; tλ 1 /θ k &lt; v , for any 0 &lt; u &lt; v &lt; + ∞ . Then, for terms with tλ 1 /θ k &lt; u we use the uniform boundedness of F V ( r, λ ) , i.e. the part F V ( r, λ ) ≤ C of statement 2, and show that their contribution can be made arbitrarily small by decreasing u . Finally, for terms with tλ 1 /θ k &gt; v we use the part F V ( r, λ ) ≤ C r of statement 2, and show that their contribution can be made arbitrarily small by increasing v .

This completes the proof of Theorem 3.

## D Proof of Proposition 1

To simplify notation, set A = 1 ; results for general A 's are easily obtained by rescaling.

Note first that for any µ ∈ C \ [0 , 1] the integral in Eq. (28) converges and is nonzero. To see that 729 it is nonzero, note that if µ has a nonzero imaginary part, then the integral has a nonzero imaginary 730 part of the opposite sign, hence is nonzero. On the other hand, if µ &gt; 1 or µ &lt; 0 , then the integral 731 is strictly positive or negative, so also nonzero. It follows that the expression in parentheses is 732 invertible and so Ψ( µ ) is well-defined for all µ ∈ C \ [0 , 1] . 733

The asymptotics Ψ( µ ) = -µ (1 + o (1)) at µ →∞ is obvious. 734

To find the asymptotics at µ → 1 , make the substitution z = δ/ ( µ -1) in the integral: 735

<!-- formula-not-decoded -->

As µ → 1 the last integral converges to a standard integral: 736

<!-- formula-not-decoded -->

The integration line in the last integral is any line connecting 0 to ∞ in C \ ( -∞ , 0); the integral 737 does not depend on the line thanks to the condition θ &gt; 1 . 738

̸

We prove now that Ψ( {| µ | ≥ 1 } ) ∩ (0 , 2] = ∅ . Let us first show that if | µ | ≥ 1 and ℑ µ = 0 , then 739 Ψ( µ ) / ∈ (0 , + ∞ ) . To this end write 740

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where J ( µ ) = µ + 1 µ is Zhukovsky's function. 741

Suppose, for definiteness, that ℑ µ &gt; 0 . Regarding a , note that if ℑ µ &gt; 0 , then the imaginary part 742 of the integrand in Eq. (210) is also positive, and so ℑ a &gt; 0 . 743

We see that Ψ( µ ) can be real and positive only if µ ∈ R . Clearly, Ψ( µ ) &gt; 0 if µ ≤ -1 , and 747 Ψ( µ ) ≤ 0 if µ ≥ 1 . It is easily checked by differentiation that Ψ( µ ) is monotone decreasing for 748 µ ∈ ( -∞ , -1] , so the smallest positive value attained by Ψ is 749

Regarding b , recall that if ℑ µ &gt; 0 and | µ | &gt; 1 , then ℑ J ( µ ) &gt; 0 . On the other hand, if | µ | = 1 , 744 then J ( µ ) ∈ [ -2 , 2] . Combining these observations, we see that if ℑ µ &gt; 0 and | µ | ≥ 1 , then either 745 ℑ b &gt; 0 , or b ≤ 0 . Since ℑ a &gt; 0 , it follows that ab / ∈ (0 , + ∞ ) . 746

<!-- formula-not-decoded -->

## E Proof of Proposition 2 750

In terms of α, b , c , D, the components P, Q of the characteristic polynomial det( µ -S λ ) = P ( µ ) -751 λQ ( µ ) can be written as 752

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(see Theorem 1 in [29]). Accordingly, 753

<!-- formula-not-decoded -->

If D = diag( d 1 , . . . , d M ) , then 754

<!-- formula-not-decoded -->

On the other hand, our definition of Ψ ( M ) implies that 755

<!-- formula-not-decoded -->

By comparing this expansion with Eq. (216), we see that the values of α, b , c , D given in Eqs. 756 (31)-(34) ensure that P/Q = Ψ ( M ) . 757

758

759

760

761

762

763

764

765

## F Experiments

The experiments in this section 2 are performed with Corner SGD approximated as in Proposition 2 with memory size M = 5 and spacing parameter l = 5 . Experiments have been performed with GPU NVIDIA GeForce RTX 4070, CPU Intel Core i5-12400F, and 32 GB RAM; the training of all the models on GPU has taken less than half an hour.

A synthetic indicator problem. Suppose that we are fitting the indicator function y ( x ) = 1 [ 1 / 4 , 3 / 4 ] ( x ) on the segment [0 , 1] using the shallow ReLU neural network in which only the output layer weights w n are trained:

<!-- formula-not-decoded -->

This is an exactly linear model that in the limit N →∞ acquires the form 766

767

768

<!-- formula-not-decoded -->

where x , w are understood as vectors in L 2 ([0 , 1]) , and x ≡ ( x -· ) + . We consider the loss L ( w ) = E x ∼ U (0 , 1) 1 2 ( x T w -y ( x )) 2 , where U (0 , 1) is the uniform distribution on [0 , 1] .

769

770

771

772

773

774

775

776

777

This limiting integral problem obeys asymptotic spectral power laws (11),(12) with precisely computable ν, ζ (see Appendix H):

<!-- formula-not-decoded -->

The problem thus falls into the sub-phase I 'full acceleration' of the signal dominated phase, and we expect that it can be accelerated with corner algorithms up to θ max = 2 .

In the experiment we set N = 10 5 and apply corner SGD with θ = 1 . 8 , see Figure 3. The experimental exponent of plain SGD is close to the theoretical value ζ = 0 . 25 . The accelerated exponent of approximate Corner SGD is slightly lower, but close to the theoretical value θζ = 1 . 8 · 0 . 25 = 0 . 45 .

MNIST. We consider MNIST [15] digit classification performed by a single-hidden-layer ReLU neural network:

<!-- formula-not-decoded -->

Here, the input vector x = ( x m ) 28 × 28 m =1 represents a MNIST image, and the outputs y r represent 778 the 10 classes. We use the one-hot encoding for the targets y ( x ) and the quadratic pointwise loss 779 ℓ ( x , w ) = 1 2 | ̂ y ( x , w ) -y ( x ) | for training. The trainable weights include both first- and second-layer 780 weights w (1) nm , w (2) rn . 781

2 A jupyter notebook with all experiments is provided in SM

<!-- image -->

Iteration

Iteration

Figure 5: Training loss of neural network (223) on MNIST classification with H = 1000 , with batch size | B | = 1000 ( left ) or 100 ( right ). The full color curves show the smoothed losses.

Note that the model (223) is nonlinear, but for large width H and standard independent weight 782 initialization it belongs to the approximately linear NTK regime [14]. In [26] MNIST was found to 783 have an approximate power-law spectrum with 784

<!-- formula-not-decoded -->

putting this problem in the sub-phase III 'limited by U Σ -finiteness' of the signal-dominated phase 785 (see Figure 1). Theoretically, by Theorem 4, the largest feasible acceleration in this case is θ max = ν . 786 Note, however, that this theoretical prediction relied on the infinite-dimensionality of the problem 787 and the divergence of the series ∑ ∞ t =1 t θ/ν -2 . The actual MNIST problem is finite-dimensional, so 788 its U Σ is always finite (though possibly large) and can be made &lt; 1 if | B | is large enough. This 789 suggests that corner SGD might practically be used with θ &gt; ν and possibly display acceleration 790 beyond the theoretical bound θ max = ν . Note also that with exponents (224) the signal/noise balance 791 bound 2 ζ +1 /ν ≈ 2 , i.e. it is not an obstacle for increasing the parameter θ towards 2. 792

793

794

795

796

In Figure 5 we test corner SGD with θ = 1 . 3 or 1.8 on batch sizes | B | = 1000 and 100 . The θ = 1 . 3 version shows a stable performance accelerating the plain SGD exponent ζ by a factor ∼ 1 . 5 . The θ = 1 . 8 version shows lower losses, but does not significantly improve acceleration factor 1 . 5 at | B | = 1000 and is unstable at | B | = 100 .

797

798

799

800

801

802

803

804

805

806

807

808

809

810

811

In Figure 6 we show both train and test trajectories of the loss and error rate (fraction of incorrectly classified images). The test performance is computed on the standard set of 10000 images, while the training performance is computed by averaging the training loss trajectory. We observe that, similarly to the training set performance, the test performance also improves faster with Corner SGD than with plain SGD. The instability of Corner SGD with θ = 1 . 8 and batch size 100 observed previously on the training set is also visible on the test set.

## G Additional notes and discussion

̸

Extension to SE approximation with τ 2 = 0 . The key assumption in our derivation and analysis of the contour representation and corner algorithms was the Spectrally Expressible approximation with τ 2 = 0 for the SGD moment evolution (see Eq. (6)). While the SE approximation in general was justified from several points of view in [25, 29], a natural question is how important is the condition τ 2 = 0 . This condition substantially simplifies the representation of propagators U t , V t in Eqs. (8), but does not seem to correspond to any specific natural data distribution ρ . (In contrast, the cases τ 1 = τ 2 = 1 and τ 1 = 1 , τ 2 = -1 exactly describe translation-invariant and Gaussian distributions; see [25].)

In fact, our analysis of the corner propagators U t , V t can be extended from τ 2 = 0 to general τ 2 by 812 a perturbation theory around τ 2 = 0 . In Appendix I we sketch an argument suggesting that, at least 813 for sufficiently large batch sizes | B | , Theorem 3 remains valid for general τ 2 , even with the same 814 coefficients C U , C V (i.e., the contribution from τ 2 = 0 produces only subleading terms in U t , V t ). 815

̸

816

817

818

819

820

821

822

823

824

825

826

827

828

829

830

831

832

833

834

835

836

837

838

839

Figure 6: MNIST trajectories of loss (top row) and error rate (bottom row) on train set (lighter colors) and test set (darker colors) . Left column: batch size 1000. Right column: batch size 100.

<!-- image -->

This implies, in particular, that the acceleration phase diagram in Theorem 4 and Figure 1 (right) is not only τ 1 -, but also τ 2 -independent.

Computational complexity. The main overhead of finitely-approximated corner algorithms compared to plain SGD lies in the memory requirements: if the model has W weights (i.e., dim w t = W in Eq. (1)), then a memoryM algorithm needs to additionally store MW scalars in the auxiliary vectors u t . On the other hand, the number of elementary operations (arithmetic operations and evaluations of standard elementary functions) in a single iteration of a finitely-approximated corner algorithm need not be much larger than for plain SGD.

Indeed, an iteration (1) of a memoryM algorithm consists in computing the gradient ∇ L ( w t ) and performing a linear transformation. In SGD with batch size | B | , the estimated gradient ∇ L B t ( w t ) is computed by backpropagation using ∝ | B | W operations. If Corner SGD is finitely-approximated using a diagonal matrix D as in Proposition 2, then the number of operations in the linear transformation is O ( MW ) . Accordingly, if | B | ≫ M (which should typically be the case in practice), then the computational cost of the linear transformation is negligible compared to the batch gradient estimation, and so the computational overhead of Corner SGD is negligible compared to plain SGD.

Practical and theoretical acceleration. Our MNIST experiment in Section F shows that finitelyapproximated Corner SGD developed in Section 5 can practically accelerate learning even on realistic problems that are not exactly linear. We note, however, that, in contrast to the ideal infinitememory Corner SGD of Section 4, this finitely-approximated Corner SGD does not theoretically accelerate the convergence exponent ζ as t → ∞ . (As shown in [29], this is generally impossible for stationary algorithms with finite linear memory.) Nevertheless, we expect that such an acceleration can be achieved with a suitable non-stationary approximation. In [29], an acceleration with a factor θ up to 2 -1 /ν was heuristically derived for a suitable non-stationary memory-1 SGD algorithm.

We remark also that if the model includes nonlinearities, then even the plain SGD in the signal840 dominated regime may show a complex picture of convergence rates depending on the strength of 841 the feature learning effects. In particular, [6] consider a particular model where the 'rich training' 842 regime is argued to accelerate the 'lazy training' exponent ζ by the factor 2 1+ ζ . This is different 843 from our factor θ max = min ( 2 , ν, 2 ζ +1 /ν ) due to a different acceleration mechanism. 844

845

846

847

848

849

850

851

852

853

854

855

856

857

858

859

860

861

862

863

864

865

866

867

868

869

870

## H The synthetic 1D example

Recall that in Section F we consider the synthetic 1D example in which we fit the target function y ( x ) = 1 [ 1 / 4 , 3 / 4 ] ( x ) on the segment [0 , 1] with a model that in the infinite-size limit has the integral form

<!-- formula-not-decoded -->

where x , w are understood as vectors in L 2 ([0 , 1]) , and x ≡ ( x -· ) + . We consider the loss L ( w ) = E 1 2 ( x T w -y ( x )) 2 , where ρ is the uniform distribution on [0 , 1] .

The asymptotic power-law structure of this problem can be derived either from general theory of singular operators and target functions, or from the specific eigendecomposition available in this simple 1D setting.

The eigenvalues. First observe that the operator H = E x ∼ ρ [ xx T ] in our case is the integral operator

<!-- formula-not-decoded -->

The operator has eigenvalues (see, e.g., Section A.6 of [28]) λ k = ξ -4 k , where

<!-- formula-not-decoded -->

Numerically, ξ 0 ≈ 1 . 875 so the leading eigenvalue λ 0 ≈ 0 . 0809 .

In particular, the capacity condition (11) holds with ν = 4 .

In fact, such a power-law asymptotics is a general property of integral operators with diagonal singularities of a particular order [5]. It is easily checked that the diagonal singularity of operator (226) is of order α = 3 . In dimension d the exponent ν has the general form ν = 1 + α d , which evaluates to 4 in our case d = 1 .

The eigencoefficients. To establish the source condition (12), we can invoke the general theory that says that for targets that are indicator function of smooth domains we have ζ = 1 d + α = 1 4 [26]. Alternatively, we can directly find ζ thanks to the simple structure of the problem.

A short (though not quite rigorous) argument is to observe that the exact minimizer w ∗ making the loss L ( w ) = 0 formally has the distributional form

<!-- formula-not-decoded -->

with Dirac delta δ ( x ) . This vector w ∗ has an infinite L 2 ([0 , 1]) norm, in agreement with our expectation that ζ = 1 4 &lt; 1 . The eigenfunctions of the problem can be explicitly found (Section A.6 of [28]):

<!-- formula-not-decoded -->

Then, formally, 871

<!-- formula-not-decoded -->

It follows that at small λ , denoting k ∗ ( λ ) = min { k : λ k &lt; λ } , 872

<!-- formula-not-decoded -->

implying again ζ = 1 4 . 873

Arigorous proof, avoiding Dirac deltas, can be given along the following lines. First note that in the 874 setting of loss function L ( w ) = 1 2 E x ∼ ρ ( x T w -y ( x )) 2 the vector q appearing in quadratic form (2) 875 acquires the form q = E x ∼ ρ [ y ( x ) x ] , which in our example gives 876

<!-- formula-not-decoded -->

We get from the condition Hw ∗ = q that 877

<!-- formula-not-decoded -->

The eigenfunctions can be written as 878

<!-- formula-not-decoded -->

where the last O ( e -ξ k ) is uniform in x ∈ [0 , 1] . Performing integration by parts twice with vanishing 879 boundary terms, we find that 880

<!-- formula-not-decoded -->

881

882

883

884

885

886

887

888

889

890

891

leading to e T k w ∗ ∝ ξ -3 k /λ k = ξ k , in agreement with Eq. (230).

## I Extending the proof of Theorem 3 to τ 2 = 0

̸

In this section we sketch (without much rigor) an argument suggesting that Theorem 3 remains valid under assumption of SE approximation with τ 2 = 0 at least if the batch size | B | is large enough.

̸

Recall that the assumption τ 2 = 0 was used to write the propagators U t , V t in the simple form (8). These representations led to the representations (19), (21) of U t , V t in terms of the contour map Ψ that were instrumental in proving Theorem 3. While we are not aware of a similar contour representation at τ 2 = 0 , we can expand the general τ 2 = 0 propagators in terms of the spectral components of the τ 2 = 0 propagators, and in this way reduce the study of the general case to the already analyzed special case.

Specifically, let us introduce the notation

<!-- formula-not-decoded -->

Then formula (8) for the propagator U t can be written as 892

<!-- formula-not-decoded -->

In the proof of Theorem 3 it was shown that (see Eqs. (83), (85)) 893

<!-- formula-not-decoded -->

̸

̸

Upon substituting tλ 1 /θ = r and applying the capacity condition (11), this gave the leading term in 894 U t : 895

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Now, if the SE approximation holds with τ 2 = 0 , then the propagator formulas (8) are no longer 896 valid. Instead (see [29]), the propagators can be written with the help of the linear transition opera897 tors A λ acting on ( M +1) × ( M +1) matrices Z : 898

<!-- formula-not-decoded -->

In particular, Eqs. (238), (239) get replaced by 899

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that Eq. (238) is a special case of Eq. (247) resulting at τ 2 = 0 thanks to the simple factorized 900 structure of the transformation A λ with vanishing second term. 901

Let us now write the binomial expansion of G ( t, λ ) by choosing one of the two terms on the r.h.s. 902 of Eq. (245) in each of the t -1 iterates of A λ in Eq. (247). The key observation here is that each 903 term in this binomial expansion can be written as a product of the τ 2 = 0 factors G 0 with a suitable 904 coefficient: 905

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

906

Here, 0 &lt; t 1 &lt; . . . &lt; t m &lt; t are the iterations at which the second term in Eq. (245) was chosen.

We can now apply again approximation (240) for G 0 in terms of F U , and approximate summation 907 by integration: 908

<!-- formula-not-decoded -->

where ( F 2 U ) ∗ ( m +1) is the ( m +1) -fold self-convolution of F 2 U : 909

<!-- formula-not-decoded -->

The factor λ 1 /θ in (250) results from the respective factor λ 2 in Eq. (248), the factor λ 2 /θ -2 in Eq. 910 (240), and the integration element scaling factor λ -1 /θ due to the substitution r n = t n λ 1 /θ . 911

The leading term in expansion (250) corresponds to the case τ 2 = 0 . Consider the next term, m = 1 . 912 The respective contribution to U t is 913

<!-- formula-not-decoded -->

This expression can be analyzed similarly to the leading term in Eq. (241), giving 914

<!-- formula-not-decoded -->

Note the faster decay t θ/ν -3 compared to t θ/ν -2 in the leading term. This difference results from 915 the different exponent 3 /θ on λ k . It also leads to the factor r 3 rather than r 2 in the integral. 916

The coefficient in brackets in Eq. (253) is finite unless the integral diverges. To see the convergence, 917 write 918

<!-- formula-not-decoded -->

and use the inequality r 2 -θ/ν ≤ (2( r -r 1 )) 2 -θ/ν +(2 r 1 ) 2 -θ/ν valid since 2 -θ/ν &gt; 0 : 919

<!-- formula-not-decoded -->

since F U ( r ) ∝ r -θ -1 as r →∞ by Lemma 1. 920

Next terms in expansion (250) can be analyzed similarly, but we encounter the difficulty 921 that, due to the associated factor λ m/θ in Eq. (250), they will contain the integrals 922 ∫ 0 ∞ r 2+ m ( F 2 U ) ∗ ( m +1) ( r ) dr -θ/ν that diverge for sufficiently large m . For this reason, it is conve923 nient to upper bound 924

<!-- formula-not-decoded -->

Then the contribution U ( m ) t to U t from the term m can be upper bounded by 925

<!-- formula-not-decoded -->

Using the inequality r 2 -θ/ν ≤ (( m +1)( r -r m )) 2 -θ/ν + . . . +(( m +1) r 1 ) 2 -θ/ν , the integral can 926 be bounded as 927

<!-- formula-not-decoded -->

Summarizing, the contribution of all the terms in U t other than the leading term U (0) t can be upper 928 bounded by 929

with the constant 930

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If 931

then series (262) converges, and so | U t -U (0) t | = o ( U (0) t ) , as claimed. 932

The case of the propagators V t can be treated similarly. Starting from τ 2 = 0 , denote 933

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then by Eqs. (153), (155) H 0 ( t, λ ) ≈ F 2 V ( tλ 1 /θ ) and 934

The counterpart of H 0 for general τ 2 is 935

<!-- formula-not-decoded -->

Expansion (248) gets replaced by 936

<!-- formula-not-decoded -->

and expansion (250) gets replaced by 937

<!-- formula-not-decoded -->

The factor λ m/θ can again be used to extract an extra negative power of t in the asymptotic bounds. 938 To avoid divergence of the integrals, we can use a bound 939

<!-- formula-not-decoded -->

with some sufficiently small ϵ &gt; 0 . Arguing as before, we then find that for | B | large enough the 940 contribution of all the terms m ≥ 1 is O ( t -θζ -ϵ ) , i.e. asymptotically negligible compared to the 941 leading term ∝ t -θζ . 942

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We believe that the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss limitations from various perspectives in the paper. We clearly formulate the theoretical model and the theoretical assumptions in Section 2. We discuss to which extent the assumptions are necessary or can possibly be relaxed (in particular, the spectrally-expressible approximation (6) with τ 2 = 0 ) in Section G. In Section 5 we point out that the ideal theoretical corner algorithms that we propose require infinite memory, but can be efficiently approximated but finite-memory algorithms. In Appendix F with experiments we show to which extent practical applications confirm the theoretical predictions.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

997

998

999

1000

1001

1002

1003

1004

1005

1006

1007

1008

1009

1010

1011

1012

1013

1014

1015

1016

1017

1018

1019

1020

1021

1022

1023

1024

1025

1026

1027

1028

1029

1030

1031

1032

1033

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

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Yes, we clearly describe the assumptions and provide complete proofs for all results.

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

Justification: We provide a jupyter notebook with all experiments in SM.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.

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

In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We provide open access to our jupyter notebook with experiments.

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

Justification: The optimizers (plain SGD and our Corner SGD), main parameters of our method (e.g., the angular parameter θ and memory size M ) as well as training and test details (e.g., the batch size | B | ) are described in Section F; all the other details can be found in the provided jupyter notebook.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Our work is primarily theoretical, with rigorous theorems as main results. The experiments only serve to confirm the theoretically predicted acceleration effects. We believe that these effects are quite visible in our experiments even without error bars, and error bars would only clutter the results. We do not compare methods with very close performance, which would indeed require error bars for reliability of comparison.

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

Justification: All this information is provided in Section F with experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We do not see any violation of the NeurIPS Code of Ethics in our research.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

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

Justification: Our work is primarily theoretical and mathematical. We expect that the methods we propose can benefit the theory and practice of optimization and machine learning, and in this sense have a positive societal impact. We are not aware of any possible negative societal impact.

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

Justification: We only use the standard MNIST dataset directly available in standard machine learning frameworks. The reference to the original publication is provided.

Guidelines:

- The answer NA means that the paper does not use existing assets.

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

Justification: We do not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. 1257

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

1271

1272

1273

1274

1275

1276

1277

1278

1279

1280

1281

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

## Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
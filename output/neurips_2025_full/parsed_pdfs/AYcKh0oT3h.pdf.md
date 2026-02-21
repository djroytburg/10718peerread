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

## Online Convex Optimization with Heavy Tails: Old Algorithms, New Regrets, and Applications

## Anonymous Author(s)

Affiliation Address email

## Abstract

In Online Convex Optimization (OCO), when the stochastic gradient has a finite variance, many algorithms provably work and guarantee a sublinear regret. However, limited results are known if the gradient estimate has a heavy tail, i.e., the stochastic gradient only admits a finite p -th central moment for some p ∈ (1 , 2] . Motivated by it, this work examines different old algorithms for OCO (e.g., Online Gradient Descent) in the more challenging heavy-tailed setting. Under the standard bounded domain assumption, we establish new regrets for these classical methods without any algorithmic modification. Remarkably, these regret bounds are fully optimal in all parameters (can be achieved even without knowing p ), suggesting that OCO with heavy tails can be solved effectively without any extra operation (e.g., gradient clipping). Our new results have several applications. A particularly interesting one is the first provable convergence result for nonsmooth nonconvex optimization under heavy-tailed noise without gradient clipping.

## 1 Introduction

This paper studies the online learning problem with convex losses, also known as Online Convex Optimization (OCO), a widely applicable framework that learns under streaming data [4, 10, 27, 35]. OCO has tons of implications for both designing and analyzing algorithms in different areas, for example, stochastic optimization [8, 23, 14], PAC learning [3], control theory [1, 11], etc.

In an OCO problem, a learning algorithm A would interact with the environment in T rounds, where T ∈ N can be either known or unknown. Formally, in each round round t , the learner A first decides an output x t ∈ X from a convex feasible set X ⊆ R d , then the environment reveals a convex loss function ℓ t : X → R , and A incurs a loss of ℓ t ( x t ) . After T many rounds, the quantity measuring the algorithm's performance is called regret, defined relative to any fixed competitor x ∈ X as follows:

<!-- formula-not-decoded -->

In the classical setting, instead of observing full information about ℓ t , the learner A is only guaranteed 24 to receive a subgradient ∇ ℓ t ( x t ) ∈ ∂ℓ t ( x t ) at its decision, where ∂ℓ t ( x t ) denotes the subdifferential 25 set of ℓ t at x t [33]. This turns out to be enough for our purpose of minimizing the regret, since any 26 OCO problem can be reduced to an Online Linear Optimization (OLO) instance via the inequality 27 ℓ t ( x t ) -ℓ t ( x ) ≤ ⟨∇ ℓ t ( x t ) , x t -x ⟩ , which holds due to convexity. Under the standard bounded 28 domain assumption, i.e., X has a finite diameter D , many classical algorithms, e.g., Online Gradient 29 Descent ( OGD ) [50], guarantee an optimal sublinear regret GD √ T for G -Lipschitz ℓ t . Even better, 30 in the case that computing an exact subgradient is intractable, and one could only query a stochastic 31 estimate g t satisfying E [ g t | x t ] ∈ ∂ℓ t ( x t ) , the OGD algorithm can still solve OCO effectively with 32

√

a provable ( G + σ ) D T regret bound in expectation if the stochastic noise g t - ∇ ℓ t ( x t ) has a 33 bounded second moment σ 2 for some σ ≥ 0 , which is called the finite variance condition. 34

35

36

37

38

39

40

41

However, many works have pointed out that even for the easier stochastic optimization (i.e., ℓ t = F for a common F ), the typical finite variance assumption is too optimistic and can be violated in different tasks [12, 37, 45], and their observations suggest that the stochastic gradient only admits a finite p -th central moment upper bounded by σ p for some p ∈ (1 , 2] , which is named heavy-tailed noise. This new assumption generalizes the classical finite variance condition ( p = 2 ) and becomes challenging when p &lt; 2 . A particular evidence is that the famous Stochastic Gradient Descent ( SGD ) algorithm [32] (which is exactly OGD for stochastic optimization) provably diverges [45].

42

43

44

45

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

Though heavy-tailed stochastic optimization has been extensively studied [18, 26, 34], limited results are known for OCO with heavy tails. The only work under this topic that we are aware of is [47], which established a parameter-free regret bound in high probability (more discussions provided later). However, their algorithm includes many nontrivial modifications like gradient clipping and significantly deviates from the existing simple OCO algorithms used in practice. Especially, consider OGD as an example. Though the heavy-tailed issue is known, OGD (or just think of it as SGD ) still works (sometimes very well) in practice even without gradient clipping and is arguably one of the most popular optimizers, which seemingly contradicts the theory of unconvergence mentioned before. This indicates that, for classical OCO algorithms under heavy-tailed noise, a huge gap exists between the empirical convergence (or even the effective practical performance) and theoretical guarantees. Therefore, we are naturally led to the following question:

In what context can old OCO algorithms work under heavy tails, in what sense, and to what extent?

## 1.1 Contributions

Motivated by the above question, we examine three classical algorithms for OCO: Online Gradient Descent ( OGD ) [50], Dual Averaging ( DA ) [25, 43], and AdaGrad [9, 22], and answer it as follows:

Under the standard bounded domain assumption, the in-expectation regret E [ R A T ( x ) ] is finite and optimal for any A ∈ { OGD , DA , AdaGrad } , without any algorithmic modification.

In detail, our new results for heavy-tailed OCO are summarized here:

- We prove the only and the first optimal regret bound E [ R A T ( x ) ] ≲ GD √ T + σDT 1 / p , ∀ x ∈ X for any A ∈ { OGD , DA , AdaGrad } . Remarkably, AdaGrad can achieve this result without knowing any of the Lipschitz parameter G , noise level σ , and tail index p .
- We extend the analysis of OGD to Online Strongly Convex Optimization with heavy tails and establish the first provable result E [ R OGD T ( x ) ] ≲ G 2 log T µ + σ p G 2 -p µ T 2 -p , ∀ x ∈ X , where µ &gt; 0 is the modulus of strong convexity and T 0 should be read as log T .

Based on the new regret bounds for OCO with heavy tails, we provide the following applications:

- For nonsmooth convex optimization with heavy tails, we show the first optimal in-expectation rate GD/ √ T + σD/T 1 -1 / p achieved without gradient clipping, which applies to both the average iterate and last iterate, demonstrating that SGD does converge once the domain is bounded.
- For nonsmooth nonconvex optimization with heavy tails, we show the first provable sample complexity of G 2 δ -1 ϵ -3 + σ p p -1 δ -1 ϵ -2 p -1 p -1 for finding a ( δ, ϵ ) -stationary point without gradient clipping. Moreover, we give the first convergence result when the problem-dependent parameters (like G , σ , and p ) are unknown in advance.

## 1.2 Discussion on [47]

As noted, [47] is the only work for OCO with heavy tails, as far as we know. There are two 75 major discrepancies between them and us. First, they consider the case where the feasible set 76 X is unbounded and aim to establish a parameter-free regret bound, i.e., the regret bound has a 77 linear dependency on ∥ x ∥ (up to an extra polylog ∥ x ∥ ) for any competitor x ∈ X . Second, they 78

focus on high-probability rather than in-expectation analysis. As such, their regret is in the form of 79

80

81

82

83

84

85

86

87

88

89

90

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

118

119

120

121

122

123

124

125

R A T ( x ) ≲ ( G + σ ) ∥ x ∥ T 1 / p , ∀ x ∈ X (up to extra polylogarithmic factors) with high probability. Without a doubt, their setting is harder than ours implying their bound is stronger as it can convert to an in-expectation regret E [ R A T ( x ) ] ≲ ( G + σ ) DT 1 / p for any bounded domain X with a diameter D .

We emphasize that the motivation behind [47] differs heavily from ours. They aim to solve heavytailed OCO with a new proposed method that contains many nontrivial technical tricks, including gradient clipping, artificially added regularization, and solving the additional fixed-point equation. However, their result cannot reflect why the existing simple OCO algorithms like OGD work in practice under heavy-tailed noise. In contrast, our goal is to examine whether, when, and how the classical OCO algorithms work under heavy tails, thereby filling the missing piece in the literature.

1 / p

Moreover, we would like to mention two drawbacks of [47]. First, though the T regret seems tight as it matches the lower bound [24, 30, 41], this may not be the best, since an optimal bound should recover the standard √ T regret in the deterministic case (i.e., σ = 0 ), as one can imagine. This suggests that their bound is not entirely optimal. Second, we remark that they require knowing both problem-dependent parameters G , σ , p and time horizon T in the algorithm, which may be hard to satisfy in the online setting. In comparison, our regret bound GD √ T + σDT 1 / p is fully optimal in all parameters. Importantly, AdaGrad can achieve it while oblivious to the problem information.

## 2 Preliminary

Notation. N denotes the set of natural numbers (excluding 0 ). [ T ] ≜ { 1 , . . . , T } , ∀ T ∈ N . a ∧ b ≜ min { a, b } and a ∨ b ≜ max { a, b } . We write a ≲ b if a ≤ Cb for a universal constant C &gt; 0 . ⌊·⌋ and ⌈·⌉ respectively represent the floor and ceiling functions. ⟨· , ·⟩ denotes the Euclidean inner product and ∥·∥ ≜ √ ⟨· , ·⟩ is the standard 2 -norm. Given x ∈ R d and D &gt; 0 , B d ( x , D ) is the Euclidean ball in R d centered at x with a radius D . In the case x = 0 , we use the shorthand B d ( D ) . Given a nonempty closed convex set A ⊆ R d , Π A is the Euclidean projection operator onto A . For a convex function f , ∂f ( x ) denotes its subgradient set at x .

Remark 1 . We choose the Euclidean norm only for simplicity. Extending the results in this work to any general norm is straightforward.

This work studies OCO in the context of Assumption 1.

Assumption 1. We consider the following series of assumptions:

- X ⊂ R d is a nonempty closed convex set bounded by D , i.e., sup x , y ∈X ∥ x -y ∥ ≤ D .
- ℓ t : X → R is convex for all t ∈ [ T ] .
- ℓ t is G -Lipschitz on X , i.e., ∥∇ ℓ t ( x ) ∥ ≤ G, ∀ x ∈ X , ∇ ℓ t ( x ) ∈ ∂ℓ t ( x ) , for all t ∈ [ T ] .
- Given a point x t ∈ X at the t -th iteration, one can query g t ∈ R d satisfying ∇ ℓ t ( x t ) ≜ E [ g t | F t -1 ] ∈ ∂ℓ t ( x t ) and E [ ∥ ϵ t ∥ p ] ≤ σ p for some p ∈ (1 , 2] and σ ≥ 0 , where F t ≜ σ ( g 1 , . . . , g t ) denotes the natural filtration and ϵ t ≜ g t -∇ ℓ t ( x t ) is the stochastic noise.

Remark 2 . D is recognized as known, like ubiquitously assumed in the OCO literature. Moreover, x t denotes the decision/output of the online learning algorithm by default.

In Assumption 1, the first three points are standard, and the fourth is the heavy-tailed noise assumption. In particular, p = 2 recovers the standard finite variance condition.

## 3 Old Algorithms under Heavy Tails

In this section, we revisit three classical algorithms for OCO: OGD , DA , and AdaGrad , whose regret bounds are well-studied in the finite variance case but remain unknown under heavy-tailed noise.

The basic idea of proving these algorithms work under heavy tails is to leverage the boundness property of X . We will describe it in more detail using OGD as an illustrated example. The analysis of DA follows a similar way at a high level, but differs in some details. However, though AdaGrad can be viewed as OGD with an adaptive stepsize, the way to utilize the boundness property is entirely different. All formal proofs are deferred to the appendix due to space limitations.

## 3.1 New Regret for Online Gradient Descent 126

## Algorithm 1 Online Gradient Descent ( OGD ) [50]

Input: initial point x 1 ∈ X , stepsize η t &gt; 0

for

t

= 1

x

t

+1

to

T

= Π

end for

We begin from arguably the most basic algorithm for OCO, Online Gradient Descent ( OGD ). 127

128

129

A well known analysis. The regret bound of OGD has been extensively studied [10, 27, 35]. The most well known analysis is perhaps the following one: for any x ∈ X , there is

<!-- formula-not-decoded -->

where the inequality holds by the nonexpansive property of Π X . Expanding both sides and rearranging 130 terms yield that 131

<!-- formula-not-decoded -->

If g t admits a finite variance, i.e., p = 2 in Assumption 1, taking expectations on both sides, then 132 following a standard analysis for η t = D ( G + σ ) √ t (or η t = D ( G + σ ) √ T if T is known) gives the regret 133

<!-- formula-not-decoded -->

However, the step of taking expectations on the R.H.S. of (1) crucially relies on the finite variance condition of g t . Therefore, one may naturally think OGD would not guarantee a finite regret if p &lt; 2 .

134

135

136

137

138

A less well known analysis 1 . As discussed, the failure of the above proof under heavy-tailed noise is due to (1). Therefore, if a tighter inequality than (1) exists, then it might be possible to show that OGD still works for p &lt; 2 . However, does it exist?

Actually, there is another less well known analysis to produce a better inequality than (1). That is, 139 first showing for any x ∈ X , by the optimality condition of the update rule, 140

<!-- formula-not-decoded -->

and then obtaining 141

<!-- formula-not-decoded -->

Note that (2) is tighter than (1) as ⟨ g t , x t -x t +1 ⟩ ≤ ∥ g t ∥ ∥ x t -x t +1 ∥ ≤ η t ∥ g t ∥ 2 2 + ∥ x t -x t +1 ∥ 2 2 η t , 142 where the first step is due to Cauchy-Schwarz inequality and the second one is by AM-GM inequality. 143

Handle p &lt; 2 in a simple way. Though we have tightened (1) into (2), can inequality (2) help to 144 overcome heavy tails? The answer is surprisingly positive, and our solution is fairly simple. Instead 145 of directly applying AM-GM inequality in the second step, we recall g t = ∇ ℓ t ( x t ) + ϵ t and use 146 triangle inequality to obtain 147

<!-- formula-not-decoded -->

On the one hand, by ∥∇ ℓ t ( x t ) ∥ ≤ G and AM-GM inequality, there is 148

<!-- formula-not-decoded -->

1 To clarify, the phrase 'less well known' is compared to the first one. This analysis itself is also well known.

X

do

(

x

t

-

η

t

g

t

)

On the other hand, let p ⋆ ≜ p p -1 and C ( p ) ≜ (4 p -4) p -1 p p , we have 149

<!-- formula-not-decoded -->

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

where ( a ) is by Young's inequality and ( b ) is due to ∥ x t -x t +1 ∥ ≤ D , p ⋆ = p p -1 , and C ( p ) = (4 p -4) p -1 p p . Next, we plug (4) and (5) back into (3), then combine with (2) to know

<!-- formula-not-decoded -->

Notably, the term ∥ ϵ t ∥ p has a correct exponent p . Thus, we can safely take expectations on both sides. Finally, a standard analysis yields the following Theorem 1 (see Appendix A for a formal proof).

Theorem 1. Under Assumption 1, taking η t = D G √ t ∧ D σt 1 / p in OGD (Algorithm 1), we have

<!-- formula-not-decoded -->

As far as we know, Theorem 1 is the first and the only provable result for OGD under heavy tails. Remarkably, it is not only tight in T [24, 30, 41] but also fully optimal in all parameters, in contrast to the bound ( G + σ ) DT 1 / p of [47]. This reveals that OCO with heavy tails can be optimally solved as effectively as the finite variance case once the domain is bounded, a classical condition adapted in many existing works.

Strongly convex functions. We highlight that the above idea can also be applied to Online Strongly Convex Optimization and leads to a sublinear regret T 2 -p better than T 1 / p . This extension can be found in Appendix A.

## 3.2 New Regret for Dual Averaging

Algorithm 2 Dual Averaging ( DA ) [25, 43]

Input:

x

1

for

t

initial point

= 1

to

T

do

x

t

+1

= Π

end for

∈ X

t

t

s

, stepsize

=1

Remark 3 . It is known that DA is a special realization of the more general Follow-the-RegularizedLeader ( FTRL ) framework [21]. To keep the work concise, we only focus on DA . The key idea to prove Theorem 2 can directly extend to show new regret for FTRL under heavy-tailed noise.

We turn our attention to the second candidate, the Dual Averaging ( DA ) algorithm, which is given in Algorithm 2. Though DA coincides with OGD when X = R d and η t = η , these two methods in general are not equivalent and can have significant performance differences in practice. Therefore, it is also important to understand DA under heavy tails.

Despite the proof strategies for OGD and DA are in different flavors (even for p = 2 ), the basic idea presented before for OGD still works here, i.e., apply the boundness property of X to make the term

∥ ϵ t ∥ have a correct exponent. Armed with this thought, we can prove the following new regret bound for DA under heavy-tailed noise. We refer the reader to Appendix B for its proof.

Theorem 2. Under Assumption 1, taking η t = D G √ t ∧ D σt 1 / p in DA (Algorithm 2), we have

<!-- formula-not-decoded -->

As far as we know, Theorem 2 is the first provable and optimal regret for DA under heavy tails. It 176 guarantees the same tight bound as in Theorem 1 up to different constants. 177

g

s

)

∑

X

(

x

1

-

η

t

&gt;

0

η

## 3.3 New Regret for AdaGrad 178

```
Algorithm 3 AdaGrad [9, 22] Input: initial point x 1 ∈ X , stepsize η > 0 for t = 1 to T do η t = ηV -1 / 2 t where V t = ∑ t s =1 ∥ g s ∥ 2 x t +1 = Π X ( x t -η t g t ) end for
```

Remark 4 . Algorithm 3 is also named AdaGrad -Norm (e.g., [42]). We simply call it AdaGrad . It is 179 straightforward to generalize Theorem 3 below to the per-coordinate update version. 180

Although Theorems 1 and 2 are optimal, they both suffer from an undesired point. That is, the 181 stepsize η t = D G √ t ∧ D σt 1 / p requires knowing all problem-dependent parameters. However, it may not 182 be easy to obtain them in an online setting. Especially, it heavily depends on the prior information 183 about the tail index p , which is hard to know (even approximately) in advance. In other words, they 184 both lack the adaptive property to an unknown environment. 185

To handle this issue, we consider AdaGrad , a classical adaptive algorithm for OCO. As can be seen, 186 AdaGrad is just OGD with an adaptive stepsize. However, it is this adaptive stepsize that can help us 187 to overcome the above undesired point. 188

189

190

191

192

193

194

195

196

Theorem 3. Under Assumption 1, taking η = D/ 2 in AdaGrad (Algorithm 3), we have

√

<!-- formula-not-decoded -->

Remark 5 . We also establish a similar result for DA with an adaptive stepsize. See Theorem 7 in Appendix B for details.

Theorem 3 provides the first regret bound for AdaGrad under heavy tails. Impressively, it is optimal even without knowing any of G , σ , and p . This surprising result once again demonstrates the power of the adaptive method, indicating it is robust to an unknown environment and even heavy-tailed noise, which may partially explain the favorable performance of many adaptive optimizers designed based on AdaGrad like RMSProp [40] and Adam [14].

We point out that the key to establishing Theorem 3 differs from the idea used before for OGD and 197 DA . Actually, Theorem 3 can be obtained in an embarrassingly simple way. It is known that AdaGrad 198 with η = D/ √ 2 on a bounded domain guarantees the following path-wise regret 199

<!-- formula-not-decoded -->

Observe that √ ∑ T t =1 ∥ g t ∥ 2 ≲ √ ∑ T t =1 ∥∇ ℓ t ( x t ) ∥ 2 + √ ∑ T t =1 ∥ ϵ t ∥ 2 ≤ G √ T + ( ∑ T t =1 ∥ ϵ t ∥ p ) 1 p , where the last step is due to ∥·∥ 2 ≤ ∥·∥ p for any p ∈ [1 , 2] . After taking expectations on both sides of (7) and applying Hölder's inequality to obtain E [ ( ∑ T t =1 ∥ ϵ t ∥ p ) 1 p ] ≤ ( ∑ T t =1 E [ ∥ ϵ t ∥ p ] ) 1 p ≤ σT 1 p , we conclude Theorem 3. To make the work self-consistent, we produce the formal proof of Theorem 3 in Appendix C.

## 4 Applications

We provide some applications based on the new regret bounds established in Section 3. The basic problem we study is optimizing a single objective F , which could be either convex or nonconvex.

## 4.1 Nonsmooth Convex Optimization

200

201

202

203

204

205

206

207

208

In this section, we consider nonsmooth convex optimization with heavy tails. 209

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

239

240

241

242

243

244

245

246

247

Convergence of the average iterate. First, we focus on convergence in average. By the classical online-to-batch conversion [3], the following corollary immediately holds.

Corollary 1. Under Assumption 1 for ℓ t ( x ) = ⟨∇ F ( x t ) , x ⟩ and let ¯ x T ≜ 1 T ∑ T t =1 x t , for any A ∈ { OGD , DA , AdaGrad } , we have

<!-- formula-not-decoded -->

Proof. By convexity, F (¯ x T ) -F ( x ) ≤ ∑ T t =1 F ( x t ) -F ( x ) T ≤ R A T ( x ) T is valid for any OCO algorithm A . We conclude from invoking Theorems 1, 2 and 3.

To the best of our knowledge, Corollary 1 gives the first and optimal convergence rate for these three algorithms in stochastic optimization with heavy tails. Especially, it implies that once the domain is bounded, the widely implemented SGD algorithm provably converges under heavy-tailed noise without any algorithmic change considered in many prior works, e.g., gradient clipping [18, 26].

We are only aware of two works [19, 41] based on Stochastic Mirror Descent ( SMD ) [24] that gave convergence results without clipping. However, they share a common drawback, i.e., their bounds are both in the form of ( G + σ ) D/T 1 -1 / p , which cannot recover the optimal rate GD/ √ T when σ = 0 .

Lastly, we highlight that for A = AdaGrad , Corollary 1 is not only optimal but also adaptive to the tail index p . As far as we know, no result has achieved this property before. This once again evidences the benefit of adaptive gradient methods.

Convergence of the last iterate. Next, we consider the more challenging last-iterate convergence, which has a long history in stochastic optimization and fruitful results in the case of p = 2 (see, e.g., [28, 36, 49]). However, less is known about heavy-tailed problems. So far, only two works [19, 29] have established the last-iterate convergence. The former is based on SMD , and the latter employs gradient clipping in SGD . Unfortunately, their rates are both in the suboptimal order ( G + σ ) D/T 1 -1 / p .

We will provide an optimal last-iterate rate based on the following lemma, which reduces the last-iterate convergence to an online learning problem.

Lemma 1 (Theorem 1 of [7]) . Suppose x 1 , . . . , x T and y 1 , . . . , y T are two sequences of vectors satisfying x t ∈ X , x 1 = y 1 and

<!-- formula-not-decoded -->

Given a convex function F ( x ) , let ℓ t ( x ) = ⟨∇ F ( y t ) , x ⟩ . Then for any online learner A , we have

<!-- formula-not-decoded -->

We emphasize that the stochastic gradient g t received by A is an estimate of ∇ F ( y t ) instead of ∇ F ( x t ) . This flexibility is due to the generality of the OCO framework. Moreover, for OGD , suppose there is no projection step, then (8) is equivalent to y t +1 = y t -T -t T η t g t , which can be viewed as SGD with a stepsize T -t T η t . For proof of Lemma 1, we refer the interested reader to [7].

Corollary 2. Under Assumption 1 for ℓ t ( x ) = ⟨∇ F ( y t ) , x ⟩ , where y t satisfies (8), for any A ∈ { OGD , DA , AdaGrad } , we have

<!-- formula-not-decoded -->

Proof.

Combine Lemma 1 and Theorems 1, 2 and 3 to conclude.

As far as we know, Corollary 2 is the first optimal last-iterate convergence rate for stochastic convex optimization with heavy tails, closing the gap in existing works.

One may notice that y t itself is not the decision made by the online learner and naturally may ask whether x t ensures the last-iterate convergence if we simply pick ℓ t = F . The answer turns out to

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

281

be positive at least for OGD (which is equivalent to SGD now). However, to prove this result, we rely on a technique specialized to stochastic optimization recently developed by [19, 44]. To not diverge from the topic of OCO, we defer the last-iterate convergence of OGD to Appendix D, in which Theorem 8 gives a general result for any stepsize η t and Corollary 4 shows the last-iterate rate under the same stepsize η t = D G √ t ∧ D σt 1 / p as in Theorem 1 before.

## 4.2 Nonsmooth Nonconvex Optimization

This section contains another application, nonsmooth nonconvex optimization with heavy tails. Due to limited space, we will provide only the necessary background. For more details, we refer the reader to [6, 13, 15, 16, 38, 39] for recent progress. We start with a new set of conditions.

Assumption 2. We consider the following series of assumptions:

- The objective F is lower bounded by F ⋆ ≜ inf x ∈ R d F ( x ) ∈ R .
- F is differentiable and well-behaved, i.e., F ( x ) -F ( y ) = ∫ 1 0 ⟨∇ F ( y + t ( x -y )) , x -y ⟩ d t .
- F is G -Lipschitz on R d , i.e., ∥∇ F ( x ) ∥ ≤ G, ∀ x ∈ R d .
- Given z t ∈ R d at the t -th iteration, one can query g t ∈ R d satisfying E [ g t | F t -1 ] = ∇ F ( z t ) and E [ ∥ ϵ t ∥ p ] ≤ σ p for some p ∈ (1 , 2] and σ ≥ 0 , where F t denotes the natural filtration and ϵ t ≜ g t -∇ F ( z t ) is the stochastic noise.

Remark 6 . The second point is a mild regularity condition introduced by [5] and becomes standard in the literature [2, 17, 48]. See Definition 1 and Proposition 2 of [5] for more details. In the fourth point, we use the same notation z t as in the algorithm being studied later. In fact, it can be arbitrary.

In nonsmooth nonconvex optimization, we aim to find a ( δ, ϵ ) -stationary point [46] (see the formal Definition 2 in Appendix E). This goal can be reduced to finding a point x ∈ R d such that ∥∇ F ( x ) ∥ δ ≤ ϵ , where ∥∇ F ( x ) ∥ δ is a quantity introduced by [5] as follows.

Definition 1 (Definition 5 of [5]) . Given a point x ∈ R d , a number δ &gt; 0 and an almost-everywhere differentiable function F , define ∥∇ F ( x ) ∥ δ ≜ inf S ⊂B ( x ,δ ) , 1 | S | ∑ y ∈ S y = x ∥ ∥ ∥ 1 | S | ∑ y ∈ S ∇ F ( y ) ∥ ∥ ∥ .

The only existing sample complexity under Assumption 2 is ( G + σ ) p p -1 δ -1 ϵ -2 p -1 p -1 in high probability [17], where we only report the dominant term and hide the dependency on the failure probability.

However, on the theoretical side, their result cannot recover the optimal bound G 2 δ -1 ϵ -3 [5] in the deterministic case. On the practical side, their method also employs the gradient clipping step, which introduces a new clipping parameter to tune. In fact, as stated in their Section 5, they observed in experiments that their algorithm without the clipping operation (exactly the algorithm we study next) still works under heavy tails. In addition, in their Section 6, they also explicitly ask whether the requirement to know G and A can be removed.

As will be seen later, we can address these points with the new regret bounds presented before.

## 4.2.1 Online-to-Nonconvex Conversion under Heavy Tails

```
Algorithm 4 Online-to-Nonconvex Conversion ( O2NC ) [5] Input: initial point y 0 ∈ R d , K ∈ N , T ∈ N , online learning algorithm A . for n = 1 to KT do Receive x n from A y n = y n -1 + x n z n = y n -1 + s n x n where s n ∼ Uniform [0 , 1] i.i.d. Query a stochastic gradient g n at z n Send g n to A end for
```

Remark 7 . Note that O2NC is a randomized algorithm. Therefore, the definition of the natural 282 filtration is adjusted to F n ≜ σ ( s 1 , g 1 , . . . , s n , g n , s n +1 ) accordingly. 283

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

305

306

307

308

309

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

We provide the Online-to-Nonconvex Conversion ( O2NC ) framework in Algorithm 4, which serves as a meta algorithm. Roughly speaking, Algorithm 4 reduces a nonconvex optimization problem to an OCO (in fact, OLO) problem, for which the K -shifting regret (see (9)) of the online learner A crucially affects the final convergence rate. However, the existing Theorem 8 of [5], a general convergence result for the above reduction, cannot directly apply to heavy-tailed noise, since its proof relies on the finite variance condition on g n (see Appendix E for more details).

Theorem 4. Under Assumption 2 and let v k ≜ -D ∑ kT n =( k -1) T +1 ∇ F ( z n ) ∥ ∑ kT n =( k -1) T +1 ∇ F ( z n ) ∥ , ∀ k ∈ [ K ] for arbitrary D &gt; 0 , then for any online learning algorithm A in O2NC (Algorithm 4), we have

<!-- formula-not-decoded -->

R A T ( v 1 , · · · , v K ) in Theorem 4 is called K -shifting regret [5], defined as follows:

<!-- formula-not-decoded -->

Theorem 4 here provides a new and the first theoretical guarantee for O2NC under heavy tails. Especially, it recovers Theorem 8 of [5] when p = 2 . A remarkable point is that the O2NC algorithm itself does not need any information about p . The proof of Theorem 4 can be found in Appendix E.

## 4.2.2 Convergence Rates

Theorem 4 enables us to apply the results presented in Section 3. Concretely, for X = B d ( D ) and any A ∈ { OGD , DA , AdaGrad } , if we reset the stepsize in A after every T iterations, there will be E [ R A T ( v 1 , · · · , v K ) ] ≲ GDK √ T + σDKT 1 / p by our new regret bounds, since v k ∈ X . With a carefully picked D , we obtain the following Theorem 5. Its proof is deferred to Appendix E.

Theorem 5. Under Assumption 2 and let ∆ ≜ F ( y 0 ) -F ⋆ and ¯ z k ≜ 1 T ∑ kT n =( k -1) T +1 z n , ∀ k ∈ [ K ] , setting any A ∈ { OGD , DA , AdaGrad } in O2NC (Algorithm 4) with a domain X = B d ( D ) for D = δ/T and resetting the stepsize in A after every T iterations, we have

<!-- formula-not-decoded -->

Notably, this is the first time confirming that gradient clipping is indeed unnecessary for the O2NC framework, matching the experimental observation of [17].

Corollary 3. Under the same setting of Theorem 5, suppose we have N ≥ 2 stochastic gradient budgets, taking K = ⌊ N/T ⌋ and T = ⌈ N/ 2 ⌉ ∧ (⌈ ( δGN/ ∆) 2 3 ⌉ ∨ ⌈ ( δσN/ ∆) p 2 p -1 ⌉) , we have

<!-- formula-not-decoded -->

Corollary 3 is obtained by optimizing K and T in Theorem 5. It implies a sample complexity of G 2 δ -1 ϵ -3 + σ p p -1 δ -1 ϵ -2 p -1 p -1 for finding a ( δ, ϵ ) -stationary point, improved over the previous bound ( G + σ ) p p -1 δ -1 ϵ -2 p -1 p -1 [17]. Furthermore, leveraging the adaptive feature of AdaGrad , Corollary 5 in Appendix E shows how to set K and T without G , σ , and p , resulting in the first provably rate for O2NC when no problem information is known in advance, which solves the problem asked by [17].

## 5 Conclusion and Limitation

This paper shows that three classical OCO algorithms, OGD , DA , and AdaGrad , can achieve the optimal in-expectation regret under heavy tails without any algorithmic modification if the feasible set is bounded, and provides some applications in stochastic optimization. The main limitation of our work is that all the proof crucially relies on the bounded domain assumption, which may not always be suitable in practice. Finding a weaker sufficient condition, under which the classical OCO algorithms work with heavy tails provably, is a direction worth studying in the future.

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

347

348

349

350

351

352

353

354

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

## References

- [1] Naman Agarwal, Brian Bullins, Elad Hazan, Sham Kakade, and Karan Singh. Online control with adversarial disturbances. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pages 111-119. PMLR, 09-15 Jun 2019. URL https://proceedings.mlr.press/v97/agarwal19c.html .
- [2] Kwangjun Ahn and Ashok Cutkosky. Adam with model exponential moving average is effective for nonconvex optimization. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems , volume 37, pages 94909-94933. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ ac8ec9b4d94c03f0af8c4fe3d5fad4fd-Paper-Conference.pdf .
- [3] N. Cesa-Bianchi, A. Conconi, and C. Gentile. On the generalization ability of on-line learning algorithms. IEEE Transactions on Information Theory , 50(9):2050-2057, 2004. doi: 10.1109/ TIT.2004.833339.
- [4] Nicolo Cesa-Bianchi and Gabor Lugosi. Prediction, Learning, and Games . Cambridge University Press, 2006.
- [5] Ashok Cutkosky, Harsh Mehta, and Francesco Orabona. Optimal stochastic non-smooth nonconvex optimization through online-to-non-convex conversion. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 6643-6670. PMLR, 23-29 Jul 2023. URL https://proceedings.mlr.press/v202/cutkosky23a.html .
- [6] Damek Davis, Dmitriy Drusvyatskiy, Yin Tat Lee, Swati Padmanabhan, and Guanghao Ye. A gradient sampling method with complexity guarantees for lipschitz functions in high and low dimensions. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems , volume 35, pages 6692-6703. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/2c8d9636f74d0207ff4f65956010f450-Paper-Conference.pdf .
- [7] Aaron Defazio, Ashok Cutkosky, Harsh Mehta, and Konstantin Mishchenko. Optimal linear decay learning rate schedules and further refinements. arXiv preprint arXiv:2310.07831 , 2023.
- [8] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research , 12(61):2121-2159, 2011. URL http://jmlr.org/papers/v12/duchi11a.html .
- [9] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research , 12(7), 2011.
- [10] Elad Hazan. Introduction to online convex optimization. Foundations and Trends® in Optimization , 2(3-4):157-325, 2016. ISSN 2167-3888. doi: 10.1561/2400000013. URL http://dx.doi.org/10.1561/2400000013 .
- [11] Elad Hazan and Karan Singh. Introduction to online control, 2025. URL https://arxiv. org/abs/2211.09619 .
- [12] Liam Hodgkinson and Michael Mahoney. Multiplicative noise and heavy tails in stochastic optimization. In International Conference on Machine Learning , pages 4262-4274. PMLR, 2021.
- [13] Michael Jordan, Guy Kornowski, Tianyi Lin, Ohad Shamir, and Manolis Zampetakis. Deterministic nonsmooth nonconvex optimization. In Gergely Neu and Lorenzo Rosasco, editors, Proceedings of Thirty Sixth Conference on Learning Theory , volume 195 of Proceedings of Machine Learning Research , pages 4570-4597. PMLR, 12-15 Jul 2023. URL https://proceedings.mlr.press/v195/jordan23a.html .

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

392

393

394

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

- [14] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014.
- [15] Guy Kornowski and Ohad Shamir. Oracle complexity in nonsmooth nonconvex optimization. Journal of Machine Learning Research , 23(314):1-44, 2022. URL http://jmlr.org/ papers/v23/21-1507.html .
- [16] Guy Kornowski and Ohad Shamir. On the complexity of finding small subgradients in nonsmooth optimization. In OPT 2022: Optimization for Machine Learning (NeurIPS 2022 Workshop) , 2022. URL https://openreview.net/forum?id=SaRQ4oTqWbP .
- [17] Langqi Liu, Yibo Wang, and Lijun Zhang. High-probability bound for non-smooth non-convex stochastic optimization with heavy tails. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 32122-32138. PMLR, 21-27 Jul 2024. URL https: //proceedings.mlr.press/v235/liu24bo.html .
- [18] Zijian Liu and Zhengyuan Zhou. Stochastic nonsmooth convex optimization with heavy-tailed noises: High-probability bound, in-expectation rate and initial distance adaptation. arXiv preprint arXiv:2303.12277 , 2023.
- [19] Zijian Liu and Zhengyuan Zhou. Revisiting the last-iterate convergence of stochastic gradient methods. In The Twelfth International Conference on Learning Representations , 2024. URL https://openreview.net/forum?id=xxaEhwC1I4 .
- [20] Zijian Liu and Zhengyuan Zhou. Nonconvex stochastic optimization under heavy-tailed noises: Optimal convergence without gradient clipping. In The Thirteenth International Conference on Learning Representations , 2025. URL https://openreview.net/forum?id=NKotdPUc3L .
- [21] Brendan McMahan. Follow-the-regularized-leader and mirror descent: Equivalence theorems and l1 regularization. In Geoffrey Gordon, David Dunson, and Miroslav Dudík, editors, Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics , volume 15 of Proceedings of Machine Learning Research , pages 525-533, Fort Lauderdale, FL, USA, 11-13 Apr 2011. PMLR. URL https://proceedings.mlr.press/ v15/mcmahan11b.html .
- [22] H Brendan McMahan and Matthew Streeter. Adaptive bound optimization for online convex optimization. arXiv preprint arXiv:1002.4908 , 2010.
- [23] H. Brendan McMahan and Matthew J. Streeter. Adaptive bound optimization for online convex optimization. In Conference on Learning Theory (COLT) , pages 244-256. Omnipress, 2010.
- [24] Arkadi Nemirovski and David Yudin. Problem complexity and method efficiency in optimization. Wiley-Interscience , 1983.
- [25] Yurii Nesterov. Primal-dual subgradient methods for convex problems. Mathematical programming , 120(1):221-259, 2009.
- [26] Ta Duy Nguyen, Thien H Nguyen, Alina Ene, and Huy Nguyen. Improved convergence in high probability of clipped gradient methods with heavy tailed noise. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems , volume 36, pages 24191-24222. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 4c454d34f3a4c8d6b4ca85a918e5d7ba-Paper-Conference.pdf .
- [27] Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213 , 2019.
- [28] Francesco Orabona. Last iterate of sgd converges (even in unbounded domains). 2020. URL https://parameterfree.com/2020/08/07/ last-iterate-of-sgd-converges-even-in-unbounded-domains/ .

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

- [29] Daniela Angela Parletta, Andrea Paudice, and Saverio Salzo. An improved analysis of the clipped stochastic subgradient method under heavy-tailed noise, 2025. URL https://arxiv. org/abs/2410.00573 .
- [30] Maxim Raginsky and Alexander Rakhlin. Information complexity of black-box convex optimization: A new look via feedback information theory. In 2009 47th Annual Allerton Conference on Communication, Control, and Computing (Allerton) , pages 803-510, 2009. doi: 10.1109/ALLERTON.2009.5394945.
- [31] Alexander Rakhlin, Ohad Shamir, and Karthik Sridharan. Making gradient descent optimal for strongly convex stochastic optimization. arXiv preprint arXiv:1109.5647 , 2011.
- [32] Herbert Robbins and Sutton Monro. A Stochastic Approximation Method. The Annals of Mathematical Statistics , 22(3):400 - 407, 1951. doi: 10.1214/aoms/1177729586. URL https: //doi.org/10.1214/aoms/1177729586 .
- [33] R Tyrrell Rockafellar. Convex analysis , volume 28. Princeton university press, 1997.
- [34] Abdurakhmon Sadiev, Marina Danilova, Eduard Gorbunov, Samuel Horváth, Gauthier Gidel, Pavel Dvurechensky, Alexander Gasnikov, and Peter Richtárik. High-probability bounds for stochastic optimization and variational inequalities: the case of unbounded variance. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 29563-29648. PMLR, 23-29 Jul 2023. URL https://proceedings.mlr.press/v202/sadiev23a.html .
- [35] Shai Shalev-Shwartz. Online learning and online convex optimization. Foundations and Trends® in Machine Learning , 4(2):107-194, 2012. ISSN 1935-8237. doi: 10.1561/2200000018. URL http://dx.doi.org/10.1561/2200000018 .
- [36] Ohad Shamir and Tong Zhang. Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. In Sanjoy Dasgupta and David McAllester, editors, Proceedings of the 30th International Conference on Machine Learning , volume 28 of Proceedings of Machine Learning Research , pages 71-79, Atlanta, Georgia, USA, 17-19 Jun 2013. PMLR. URL https://proceedings.mlr.press/v28/shamir13.html .
- [37] Umut Simsekli, Levent Sagun, and Mert Gurbuzbalaban. A tail-index analysis of stochastic gradient noise in deep neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pages 5827-5837. PMLR, 09-15 Jun 2019. URL https://proceedings.mlr.press/v97/simsekli19a.html .
- [38] Lai Tian and Anthony Man-Cho So. No dimension-free deterministic algorithm computes approximate stationarities of lipschitzians. Mathematical Programming , 208(1):51-74, 2024.
- [39] Lai Tian, Kaiwen Zhou, and Anthony Man-Cho So. On the finite-time complexity and practical computation of approximate stationarity concepts of Lipschitz functions. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 21360-21379. PMLR, 17-23 Jul 2022. URL https://proceedings.mlr.press/v162/tian22a.html .
- [40] Tijmen Tieleman, Geoffrey Hinton, et al. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning , 4(2): 26-31, 2012.
- [41] Nuri Mert Vural, Lu Yu, Krishna Balasubramanian, Stanislav Volgushev, and Murat A Erdogdu. Mirror descent strikes again: Optimal stochastic convex optimization under infinite noise variance. In Po-Ling Loh and Maxim Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory , volume 178 of Proceedings of Machine Learning Research , pages 65-102. PMLR, 02-05 Jul 2022. URL https://proceedings.mlr.press/v178/vural22a.html .

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

497

498

499

500

501

502

503

504

505

- [42] Rachel Ward, Xiaoxia Wu, and Leon Bottou. AdaGrad stepsizes: Sharp convergence over nonconvex landscapes. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pages 6677-6686. PMLR, 09-15 Jun 2019. URL https:// proceedings.mlr.press/v97/ward19a.html .
- [43] Lin Xiao. Dual averaging method for regularized stochastic learning and online optimization. In Y. Bengio, D. Schuurmans, J. Lafferty, C. Williams, and A. Culotta, editors, Advances in Neural Information Processing Systems , volume 22. Curran Associates, Inc., 2009. URL https://proceedings.neurips.cc/paper\_files/paper/2009/file/ 7cce53cf90577442771720a370c3c723-Paper.pdf .
- [44] Moslem Zamani and François Glineur. Exact convergence rate of the last iterate in subgradient methods. arXiv preprint arXiv:2307.11134 , 2023.
- [45] Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, Sanjiv Kumar, and Suvrit Sra. Why are adaptive methods good for attention models? In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 15383-15393. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/ b05b57f6add810d3b7490866d74c0053-Paper.pdf .
- [46] Jingzhao Zhang, Hongzhou Lin, Stefanie Jegelka, Suvrit Sra, and Ali Jadbabaie. Complexity of finding stationary points of nonconvex nonsmooth functions. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 11173-11182. PMLR, 13-18 Jul 2020. URL https://proceedings.mlr.press/v119/zhang20p.html .
- [47] Jiujia Zhang and Ashok Cutkosky. Parameter-free regret in high probability with heavy tails. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems , volume 35, pages 8000-8012. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ 349956dee974cfdcbbb2d06afad5dd4a-Paper-Conference.pdf .
- [48] Qinzi Zhang and Ashok Cutkosky. Random scaling and momentum for non-smooth non-convex optimization. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 58780-58799. PMLR, 21-27 Jul 2024. URL https://proceedings.mlr.press/ v235/zhang24k.html .
- [49] Tong Zhang. Solving large scale linear prediction problems using stochastic gradient descent algorithms. In Proceedings of the twenty-first international conference on Machine learning , page 116, 2004.
- [50] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th international conference on machine learning (icml-03) , pages 928-936, 2003.

## A Missing Proofs for Online Gradient Descent 506

This section provides missing proofs for regret bounds of OGD . Before showing the formal proof, 507 we recall the following core inequality that holds for any x ∈ X given in (6): 508

<!-- formula-not-decoded -->

The key to establishing the above result is showing 509

<!-- formula-not-decoded -->

the proof of which is by combining (3), (4), and (5) established in the main text. 510

## A.1 Proof of Theorem 1 511

Proof. For any x ∈ X , sum up (10) from t = 1 to T and drop the term -∥ x T +1 -x ∥ 2 2 η T to obtain 512

T

∑

t

=1

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step is due to ∥ x t -x ∥ ≤ D, ∀ t ∈ [ T ] and η t +1 ≤ η t , ∀ t ∈ [ T -1] . 513

Taking expectations on both sides of (13) yields that 514

<!-- formula-not-decoded -->

where for the L.H.S., we use E [ ⟨ g t , x t -x ⟩ ] = E [ E [ ⟨ g t , x t -x ⟩ | F t -1 ]] and 515

<!-- formula-not-decoded -->

for the R.H.S., we use E [ ∥ ϵ t ∥ p ] ≤ σ p . 516

Finally, we plug η t = D G √ t ∧ D σt 1 / p , ∀ t ∈ [ T ] into (14), then use ∑ T t =1 1 √ t ≲ √ T and ∑ T t =1 1 t 1 -1 / p ≲ 517 T 1 / p to conclude 518 √

519

## A.2 Extension to Online Strongly Convex Optimization 520

521

Next, we extend Theorem 1 to the strongly convex case, i.e., ∃ µ &gt; 0 such that for all t ∈ [ T ] ,

<!-- formula-not-decoded -->

In this setting, it is well known that OGD achieves a logarithmic regret bound when p = 2 [10, 27]. 522 Theorem 6 below provides the first provable result for p &lt; 2 . 523

Theorem 6. Under Assumption 1 and additionally assuming (16), taking η t = 1 µt in OGD (Algorithm 524 1), we have 525

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

⟨

g

t

,

x

t

-

x

⟩

Theorem 6 shows that under strongly convexity, OGD for p ∈ (1 , 2) achieves a better sublinear regret 526 T 2 -p than T 1 / p in Theorem 1 as 2 -p ≤ 1 / p , ∀ p &gt; 0 . One point we highlight here is that the 527 stepsize η t = 1 µt is commonly used in the OCO literature and is independent of the tail index p . 528

529

530

531

532

533

However, in contrast to Theorem 1, we suspect Theorem 6 is not tight in T for p ∈ (1 , 2) . The reason is that for nonsmooth strongly convex optimization with heavy tails (i.e., ℓ t = F, ∀ t ∈ [ T ] where F is strongly convex), Theorem 6 can convert to a convergence rate only in the order of 1 /T p -1 , which is worse than the lower bound 1 /T 2 -2 / p [45]. Therefore, we conjecture that a way to obtain a better regret bound than T 2 -p exists, which we leave as future work.

Proof of Theorem 6. For any x ∈ X , we take expectations on both sides of (12) to have 534

<!-- formula-not-decoded -->

where for the L.H.S., we follow a similar step of reasoning out (15) but instead using 535

<!-- formula-not-decoded -->

for the R.H.S., we use E [ ∥ ϵ t ∥ p ] ≤ σ p . 536

Next, we plug η t = 1 µt , ∀ t ∈ [ T ] into (17) to obtain 537

<!-- formula-not-decoded -->

Lastly, it is known that if ℓ t is G -Lipschitz and µ -strongly convex on a domain X with a diameter D , 538 then it satisfies D ≲ G µ (e.g., see Lemma 2 of [31]). Therefore, when p ∈ (1 , 2) , 539

<!-- formula-not-decoded -->

## B Missing Proofs for Dual Averaging

This section provides missing proofs for regret bounds of DA .

## B.1 Proof of Theorem 2

Proof. Let L t ( x ) ≜ ∥ x -x 1 ∥ 2 2 η t -1 + ∑ t -1 s =1 ⟨ g s , x ⟩ , ∀ t ∈ [ T +1] , where η 0 ≜ η 1 . Then DA can be equivalently written as

x

t

= argmin

By Lemma 7.1 of [27], for any x ∈ X , 546

<!-- formula-not-decoded -->

x

∈X

L

t

(

x

)

,

∀

t

∈

[

T

+1]

.

540

541

542

543

544

545

where the inequality holds by L T +1 ( x T +1 ) ≤ L T +1 ( x ) , ∀ x ∈ X due to x T +1 = 547 argmin x ∈X L T +1 ( x ) . Note that for any t ∈ [ T ] , 548

<!-- formula-not-decoded -->

where ( a ) is by η t ≤ η t -1 , ∀ t ∈ [ T ] and ( b ) is holds because L t is 1 η t -1 -strongly convex and 549 x t = argmin x ∈X L t ( x ) , which together imply 550

<!-- formula-not-decoded -->

Therefore, we have 551

<!-- formula-not-decoded -->

By the same argument as proving (11) but replacing η t with η t -1 , there is 552

<!-- formula-not-decoded -->

As such, we know 553

<!-- formula-not-decoded -->

Finally, following similar steps in proving Theorem 1 in Appendix A, we conclude 554

<!-- formula-not-decoded -->

## B.2 Dual Averaging with an Adaptive Stepsize

We show that DA with an adaptive stepsize can also achieve the optimal regret GD √ T + σDT 1 / p .

555

556

557

558

559

Theorem 7. Under Assumption 1, taking η t = 2 DV -1 / 2 t and V t = ∑ t s =1 ∥ g s ∥ 2 in DA (Algorithm 2), we have E [ R DA T ( x ) ] ≲ GD √ T + σDT 1 / p , ∀ x ∈ X .

Proof. For any x ∈ X , we have 560

<!-- formula-not-decoded -->

where η 0 ≜ η 1 . On the one hand, we can use AM-GM inequality to bound 561

<!-- formula-not-decoded -->

On the other hand, we know 562

<!-- formula-not-decoded -->

where the second step is by Cauchy-Schwarz inequality. Therefore, for any t ≥ 2 , 563

<!-- formula-not-decoded -->

where ( a ) is due to x ∧ y ≤ 2 x -1 + y -1 , ∀ x, y &gt; 0 , ( b ) is by η t -1 = 2 D √ ∑ t -1 s =1 ∥ g s ∥ 2 , and ( c ) holds 564 because of √ ∑ t s =1 ∥ g s ∥ 2 ≤ √ ∑ t -1 s =1 ∥ g s ∥ 2 + ∥ g t ∥ . Note that (21) is also true for t = 1 by (20). 565

Combine (19) and (21) and use ∥ x -x 1 ∥ ≤ D to obtain 566

<!-- formula-not-decoded -->

which only differs from (22) by a constant. Hence, by a similar proof for (24), there is 567

<!-- formula-not-decoded -->

implying 568

569

## C Missing Proofs for AdaGrad 570

This section provides missing proofs for regret bounds of AdaGrad . 571

572

## C.1 Proof of Theorem 3

Proof. As mentioned, AdaGrad can be viewed as OGD with a stepsize η t = η √ V t = η √ ∑ t s =1 ∥ g s ∥ 2 . 573 Therefore, we can use (1) for AdaGrad to know for any x ∈ X , 574

<!-- formula-not-decoded -->

Sum up the above inequality from t = 1 to T and drop the term -∥ x T +1 -x ∥ 2 2 η T to have 575

<!-- formula-not-decoded -->

where the last step is by ∥ x t -x ∥ ≤ D, ∀ t ∈ [ T ] and η t +1 ≤ η t , ∀ t ∈ [ T -1] . 576 Next, observe that for any t ∈ [ T ] , 577

<!-- formula-not-decoded -->

where 1 /η 0 should be read as 0 . The above inequality implies 578

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combine (22) and (23) to have 579

<!-- formula-not-decoded -->

Note that there is 580

<!-- formula-not-decoded -->

where the last step is due to ∥·∥ 2 ≤ ∥·∥ p for any p ∈ [1 , 2] . Hence, we obtain 581

<!-- formula-not-decoded -->

We take expectations on both sides of (24), then apply Hölder's inequality to have 582

<!-- formula-not-decoded -->

and finally plug in η = D/ 2 to conclude 583

<!-- formula-not-decoded -->

## D Missing Proofs for Applications: Nonsmooth Convex Optimization

We prove the following last-iterate convergence result for SGD (i.e., OGD for stochastic optimization) under heavy-tailed noise. The proof of Theorem 8 is inspired by [19, 44].

Theorem 8. Under Assumption 1 for ℓ t ( x ) = F ( x ) , for any stepsize η t &gt; 0 in OGD (Algorithm 1), we have

584

585

586

587

588

589

<!-- formula-not-decoded -->

Proof. Given x ∈ X , we recursively define 590

<!-- formula-not-decoded -->

in which 591

<!-- formula-not-decoded -->

Equivalently, y t can be written into a convex combination of x , x 1 , . . . , x t as 592

<!-- formula-not-decoded -->

Therefore, y t also falls into X and satisfies y t ∈ F t -1 . 593

We invoke (10) for y t to obtain 594

<!-- formula-not-decoded -->

Since x t , y t ∈ F t -1 , there is 595

<!-- formula-not-decoded -->

where the last step is due to the convexity of F . As such, we can take expectations on both sides of 596 (28) to have 597

<!-- formula-not-decoded -->

where the second step is due to ∥ x t -y t ∥ 2 ≤ ( 1 -w t -1 w t ) ∥ x t -x t ∥ 2 + w t -1 w t ∥ ∥ x t -y t -1 ∥ ∥ 2 = 598 w t -1 w t ∥ ∥ x t -y t -1 ∥ ∥ 2 by (25) and the convexity of ∥ x t -·∥ 2 . Mutiply both sides of (29) by w t η t and 599 sum up from t = 1 to T to obtain 600

<!-- formula-not-decoded -->

Now observe that 601

<!-- formula-not-decoded -->

which implies 602

<!-- formula-not-decoded -->

Thus, we can lower bound the L.H.S. of (30) by 603

<!-- formula-not-decoded -->

where the last step is due to, for t ∈ [ T -1] , 604

<!-- formula-not-decoded -->

and w T (26) = w T -1 = 1 . 605

We plug (31) back into (30) and divide both sides by w T η T to obtain 606

<!-- formula-not-decoded -->

607

608

609

610

611

612

613

Equipped with Theorem 8, we show the following anytime last-iterate convergence rate for SGD / OGD . As far as we know, this is the first and the only provable result demonstrating that the last iterate of SGD can converge in heavy-tailed stochastic optimization without gradient clipping. Compared to Corollary 2, the difference is up to an extra logarithmic factor. Therefore, it is nearly optimal.

Corollary 4. Under Assumption 1 for ℓ t ( x ) = F ( x ) , taking η t = D G √ t ∧ D σt 1 / p in OGD (Algorithm 1), we have

<!-- formula-not-decoded -->

Proof. By Theorem 8, we have 614

<!-- formula-not-decoded -->

For any t ∈ { 0 } ∪ [ T -1] , observe that by Cauchy-Schwarz inequality 615

<!-- formula-not-decoded -->

Thus, there is 616

We first bound 617

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies 618

Next, we know 619

<!-- formula-not-decoded -->

where ( a ) is by η t ≤ D G √ t and 1 η s ≤ G √ s D ∨ σs 1 / p D . Hence, there is 620

<!-- formula-not-decoded -->

Similarly, we can bound 621

<!-- formula-not-decoded -->

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

Finally, we plug (33), (34) and (35) back into (32) to conclude.

## E Missing Proofs for Applications: Nonsmooth Nonconvex Optimization

## E.1 ( δ, ϵ ) -Stationary Points

Definition 2 (Definition 4 of [5]) . Apoint x ∈ R d is a ( δ, ϵ ) -stationary point of an almost-everywhere differentiable function F if there is a finite subset S ⊂ B d ( x , δ ) such that for y selected uniformly at random from S , E [ y ] = x and ∥ E [ ∇ F ( y )] ∥ ≤ ϵ .

The concept of the ( δ, ϵ ) -stationary point presented here is due to [5], which is mildly more stringent than the notion of [46], since the latter does not require E [ y ] = x . For more discussions, see Section 2.1 of [5].

## E.2 Proof of Theorem 4

In this section, our ultimate goal is to prove Theorem 4 for the O2NC algorithm, extending Theorem 8 of [5] from p = 2 to any p ∈ (1 , 2] . Notably, our new result does not require any modification to the O2NC method, but is obtained only from a more careful analysis, indicating that O2NC is a robust and powerful algorithmic framework.

We begin with Lemma 2, which lies as the cornerstone for establishing the convergence of O2NC .

Lemma 2 (Theorem 7 of [5]) . Under Assumption 2 (only need the second point and the unbiased part in the fourth point), for any sequence of vectors u 1 , . . . , u KT ∈ R d , O2NC (Algorithm 4) guarantees

<!-- formula-not-decoded -->

To relate Lemma 2 to the concept of K -shifting regret introduced before (see (9)), suppose now a 639 sequence of vectors v 1 , . . . , v K is given, if we set u n = v k for all n ∈ { ( k -1) T +1 , . . . , kT } 640 and k ∈ [ K ] , then the second term on the R.H.S. of (36) can be written as E [ R A T ( v 1 , . . . , v K ) ] , and 641 the third term can be simplified into ∑ K k =1 E [〈 ∑ kT n =( k -1) T +1 g n , v k 〉] . 642

<!-- formula-not-decoded -->

Same as [5], we pick v k ≜ -D ∑ kT n =( k -1) T +1 ∇ F ( z n ) ∥ ∑ kT n =( k -1) T +1 ∇ F ( z n ) ∥ for some constant D &gt; 0 , which gives us 643

<!-- formula-not-decoded -->

̸

If ϵ n has a finite variance (i.e., p = 2 ), then like [5], one can invoke Hölder's inequality and use the 644 fact E [ ⟨ ϵ m , ϵ n ⟩ ] = 0 , ∀ m = n ∈ [ KT ] to obtain for any k ∈ [ K ] , 645

<!-- formula-not-decoded -->

However, this argument immediately fails when p &lt; 2 as E [ ∥ ϵ n ∥ 2 ] can be + ∞ . To handle this 646 potential issue, we require the following Lemma 3. 647

Lemma 3 (Lemma 4.3 of [20]) . Given a vector-valued martingale difference sequence w 1 , . . . , w T , 648 there is 649

650

651

<!-- formula-not-decoded -->

Equipped with Lemmas 2 and 3, we are ready to formally prove Theorem 4, demonstrating that the O2NC framework provably works under heavy-tailed noise.

652

653

Proof of Theorem 4. We invoke Lemma 2 with u n = v ⌈ n/T ⌉ , ∀ n ∈ [ KT ] (equivalently, u n = v k if n ∈ { ( k -1) T +1 , . . . , kT } ) and use the definition of K -shifting regret (see (9)) to obtain

<!-- formula-not-decoded -->

Recall that g n = ∇ F ( z n ) + ϵ n , which implies for any k ∈ [ K ] , 654

<!-- formula-not-decoded -->

where the second step is by Cauchy-Schwarz inequality and the last equation holds due to 655

<!-- formula-not-decoded -->

Combine (37) and (38), apply F ( y KT ) ≥ F ⋆ , and rearrange terms to have 656

<!-- formula-not-decoded -->

For any fixed k ∈ [ K ] , we apply Lemma 3 with w t = ϵ ( k -1) T + t , ∀ t ∈ [ T ] to know 657

<!-- formula-not-decoded -->

where the second step is by Hölder's inequality (note that p &gt; 1 ). Finally, we conclude the proof 658 after plugging (41) back into (40). 659

660

## E.3 Proof of Theorem 5

Proof. By Theorem 4, there is 661

<!-- formula-not-decoded -->

Note that A has the domain X = B d ( D ) and s n ∼ Uniform [0 , 1] . Thus, for any n ∈ [ KT ] , 662

<!-- formula-not-decoded -->

We first lower bound the L.H.S. of (42). Given k ∈ [ K ] , for any m&lt;n ∈ { ( k -1) T +1 , . . . , kT } , 663 observe that 664

<!-- formula-not-decoded -->

Recall that ¯ z k = 1 T ∑ kT n =( k -1) T +1 z n and D = δ/T now, then the above inequality implies 665

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which means 666

By the definition of ∥∇ F (¯ z k ) ∥ δ (see Definition 1), there is 667

<!-- formula-not-decoded -->

Next, we upper bound the R.H.S. of (42). By the definition of K -shifting regret (see (9)), there is 668

<!-- formula-not-decoded -->

Note that we reset the stepsize in A after every T iterations and v k ∈ B d ( D ) by its definition (see 669 (39)). Then for any A ∈ { OGD , DA , AdaGrad } , we can invoke its regret bound 2 (i.e., Theorems 1, 2 670 and 3) to obtain 671

<!-- formula-not-decoded -->

2 A minor point here is that the current function ℓ n ( x ) = ⟨ g n , x ⟩ does not entirely fit Assumption 1. We clarify that one does not need to worry about it, since all results proved in Section 3 hold under this change. For example, in the proof of Theorem 1, we can safely replace the L.H.S. of (14) with E [ ∑ T t =1 ⟨ g t , x t -x ⟩ ] .

which implies 672

<!-- formula-not-decoded -->

Finally, we plug (45) and (46) back into (42), then use D = δ/T and ∆ = F ( y 0 ) -F ⋆ to have 673

<!-- formula-not-decoded -->

## E.4 Proof of Corollary 3

Proof. Recall that we pick

674

675

676

<!-- formula-not-decoded -->

where ∆ = F ( y 0 ) -F ⋆ . We invoke Theorem 5 and use KT ≥ N/ 4 (see Fact 2) to obtain 677

<!-- formula-not-decoded -->

By the definition of T , we know 678

<!-- formula-not-decoded -->

and 679

<!-- formula-not-decoded -->

Therefore, there is

680

<!-- formula-not-decoded -->

## E.5 Extension to the Case of Unknown Problem-Dependent Parameters

In Corollary 5, we show how to set K and T when all problem-dependent parameters are unknown. It is particularly meaningful for AdaGrad . As in that case, the rate is achieved without knowing any problem-dependent parameter. This kind of result is the first to appear for nonsmooth nonconvex optimization with heavy tails. However, the rate is not as good as Corollary 3. It is currently unclear whether the same bound 1 / ( δN ) p -1 2 p -1 as in Corollary 3 can be obtained when no information about the problem is known.

Corollary 5. Under the same setting of Theorem 5, suppose we have N ≥ 2 stochastic gradient budgets, taking K = ⌊ N/T ⌋ and T = ⌈ N/ 2 ⌉ ∧ ⌈ ( δN ) 2 3 ⌉ , we have

681

682

683

684

685

686

687

688

689

690

<!-- formula-not-decoded -->

Proof. We invoke Theorem 5 and use KT ≥ N/ 4 (see Fact 2) to obtain 691

<!-- formula-not-decoded -->

By the definition of T , we know 692

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and 693

Therefore, there is 694

<!-- formula-not-decoded -->

695

## F Algebraic Facts 696

We give two useful algebraic facts in this section. 697

Fact 1. For any T ∈ N and a ∈ (0 , 1) , there is 698

<!-- formula-not-decoded -->

Proof. Note that ∑ T s = t +1 s a ≤ ( T -t ) T a , which implies 699

<!-- formula-not-decoded -->

700

Fact 2. Given 2 ≤ N ∈ N , K = ⌊ N/T ⌋ and T ∈ N satisfying T ≤ ⌈ N/ 2 ⌉ , there is KT ≥ N/ 4 . 701

702

Proof.

Note that

KT

=

⌊

N/T

⌋

T

≥

N

-

T

≥

(

N

-

1)

/

2

≥

N/

4

.

703

704

705

706

707

708

709

710

711

712

713

714

715

716

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

729

730

731

732

733

734

735

736

737

738

739

740

741

742

743

744

745

746

747

748

749

750

751

752

753

754

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims made in the abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitation in Section 5.

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

Justification: For each theoretical result, the paper provides the full set of assumptions and a complete (and correct) proof.

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

Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed because this paper is purely theoretical.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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

|   1014 | Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology,   |
|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   1015 |                                                                                                                                                                                                                                                                                 |
|   1016 |                                                                                                                                                                                                                                                                                 |
|   1017 | scientific rigorousness, or originality of the research, declaration is not required.                                                                                                                                                                                           |
|   1018 | Answer: [NA]                                                                                                                                                                                                                                                                    |
|   1019 | Justification: The core method development in this research does not involve LLMs as any                                                                                                                                                                                        |
|   1020 | important, original, or non-standard components.                                                                                                                                                                                                                                |
|   1021 | Guidelines:                                                                                                                                                                                                                                                                     |
|   1022 | • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.                                                                                                                           |
|   1023 | • Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM                                                                                                                                                                                                      |
|   1024 | )                                                                                                                                                                                                                                                                               |
|   1025 | for what should or should not be described.                                                                                                                                                                                                                                     |
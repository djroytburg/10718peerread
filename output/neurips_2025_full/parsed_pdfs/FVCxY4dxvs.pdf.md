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

## BCOS: A Method for Stochastic Approximation

## Anonymous Author(s)

Affiliation Address email

## Abstract

We consider stochastic approximation with block-coordinate stepsizes and propose adaptive stepsize rules that aim to minimize the expected distance of the next iterate from an optimal point. These stepsize rules use online estimates of the second moment of the search direction along each block coordinate, and the popular Adam algorithm can be interpreted as using a particular heuristic for such estimation. By leveraging a simple conditional estimator, we derive variants of BCOS that obtain competitive performance but require fewer optimizer states and hyper-parameters. In addition, our convergence analysis relies on a simple aiming condition that assumes neither convexity nor smoothness, thus has broad applicability.

## 1 Introduction

We consider unconstrained stochastic optimization problems of the form

<!-- formula-not-decoded -->

where x ∈ R n is the decision variable, ξ is a random variable, and f is the loss function. In the context of machine learning, x represents the parameters of a prediction model, ξ represents randomly sampled data, and f ( x, ξ ) is the loss in making predictions about ξ using the parameters x .

Suppose that for any pair x and ξ , we can evaluate the gradient of f with respect to x , denoted as ∇ f ( x, ξ ) . Starting with an initial point x 0 ∈ R n , the classical stochastic approximation method [38] generates a sequence { x 1 , x 2 , . . . } with the update rule

<!-- formula-not-decoded -->

where α t is the stepsize , which is often called the learning rate in the machine learning literature. The convergence properties of this method are well studied in the stochastic approximation literature [e.g., 38, 3, 6, 44, 52]. Despite the rich literature on their convergence theory, stochastic approximation methods in practice often require heuristics and trial and error in choosing the stepsize sequence { α t } . Adaptive rules that can adjust stepsizes on the fly have been developed in both the optimization literature [e.g., 10, 25, 33, 40, 41, 42, 43] and by the machine learning community [e.g., 22, 32, 46, 47]. More recently, adaptive algorithms that use coordinate-wise stepsizes have become very popular following the seminal works of AdaGrad [14] and Adam [26]. In this paper, we present a framework for better understanding such methods and propose a family of new, effective methods.

## 1.1 Stochastic approximation with block-coordinate stepsizes

We focus on stochastic approximation with block-coordinate stepsizes , specifically of the form

<!-- formula-not-decoded -->

where d t ∈ R n is a stochastic search direction, s t ∈ R n is a vector of coordinate-wise stepsizes, and glyph[circledot] denotes element-wise product (Hadamard product) of two vectors. The two most common

31

32

33

34

35

36

37

38

39

40

41

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

choices for the search direction are: the stochastic gradient , i.e., d t = ∇ f ( x t , ξ t ) , and its exponential moving average (EMA) . Let g t = ∇ f ( x t , ξ t ) , the EMA of stochastic gradient can be expressed as

<!-- formula-not-decoded -->

where β ∈ [0 , 1) is a smoothing factor. This is often called the stochastic momentum .

The Adam algorithm [26] uses the direction in (4) and sets the coordinate-wise stepsizes as

<!-- formula-not-decoded -->

where α t ∈ R is a common stepsize schedule and each v t,i is the EMA of the squared coordinate gradient g 2 t,i , with a different , often larger , smoothing factor β ′ ∈ (0 , 1) . More specifically,

<!-- formula-not-decoded -->

Here glyph[epsilon1] &gt; 0 is a small constant to improve numerical stability when v i,t becomes very close to zero.

Adam [26] and its variant AdamW [31] have been very successful in training large-scale deep learning models. However, theoretical understanding of their convergence properties and empirical performance is still incomplete despite a lot of recent efforts [e.g., 37, 4, 1, 9, 56, 55, 28]. On the other hand, there have been many works that propose new variants or alternatives to Adam/AdamW, either starting from fundamental principles [e.g., 53, 17, 21, 29, 24] or based on empirical algorithm search [e.g., 5, 54] But all have limited success. Adam and especially AdamW are still the dominant algorithms for training large deep learning models, and their effectiveness remains a myth.

## 1.2 Contributions and outline

We propose a family of block-coordinate optimistic stepsize (BCOS) rules for stochastic approximation. BCOS provides a novel interpretation of Adam and AdamW and their convergence analysis as special cases of a general framework. Moreover, we derive variants of BCOS that obtain competitive performance but require fewer optimizer states and hyper-parameters. More specifically:

- In Section 2, we derive BCOS by minimizing the expected distance of the next iterate from an optimal point. While the optimal stepsizes cannot be computed exactly, we make optimistic simplifications and approximate the second moment of gradients with simple EMA estimators.
- In Section 3, we instantiate BCOS with specific search directions. In particular, we show that RMSprop [48] and Adam [26] can be interpreted as special cases of BCOS. By leveraging a simple conditional estimator, we derive new variants that require fewer optimizer states and hyper-parameters. Integrating with decoupled weight decay [31] gives the BCOSW variants.
- In Section 4, we present convergence analysis of BCOS(W) based on a simple aiming condition, which assumes neither convexity nor smoothness, thus has broad applicability. We obtain strong guarantees in terms of almost sure convergence, and characterize the effect of signal-to-noise ratio of the online estimators on the convergence behavior. Our results also apply to Adam(W).
- Finally, in Section 5, we present numerical experiments to compare BCOSW and AdamW on several Deep Learning tasks and demonstrate the effectiveness of the proposed methods.

## 1.3 Notations

Let I 1 , . . . , I m be a non-overlapping partition of the coordinate index set { 1 , . . . , n } , each with cardinality n k = |I k | . Correspondingly, we partition the vectors x t , s t and d t into blocks x t,k , s t,k and d t,k in R n k for k = 1 , . . . , m . We use a common stepsize γ t,k ∈ R within each block, i.e., s t,k = γ t,k 1 n k . As a result, the explicit block-coordinate update form of (3) can be written as

<!-- formula-not-decoded -->

Notice that γ t,k is always a scalar and γ t is a vector in R m instead of R n (unless m = n ).

Throughout this paper, 〈· , ·〉 denotes the standard inner product in R n and ‖ · ‖ the induced Euclidean norm. The signum function is defined as sign ( α ) = 1 if α &gt; 0 , -1 if α &lt; 0 and 0 if α = 0 .

## 2 Derivation of BCOS

We first derive the ideal optimal stepsizes for block-coordinate update, which is not computable in 72 practice; then we make several simplifications and approximations to derive the practical ones. 73

## 2.1 Block-coordinate optimal stepsizes 74

- We consider the change of distance to an optimal point x ∗ after one iteration of the algorithm (3): 75

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Exploiting the block partitions of x t , s t and d t and using s t,k = γ t,k 1 n k , we obtain 76

<!-- formula-not-decoded -->

77

78

79

80

Taking expectation conditioned on the realization of all random variables up to x t , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In order to minimize the expected distance from x t +1 to x ∗ , we can minimize the right-hand side of (8) over the stepsizes { γ t,k } m k =1 . This results in the optimal stepsizes

<!-- formula-not-decoded -->

Notice that these optimal stepsizes can be positive or negative, depending on the sign of the inner 81 product in the numerator. Apparently, they are not computable in practice, because we do not have 82 access of x ∗ and cannot evaluate the expectations precisely. We address this issue in the next section. 83

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

## 2.2 Block-coordinate optimistic stepsizes

We need to make several simplifications and approximations to derive a practical stepsize rule. Our first step aims to avoid the direct reliance on x ∗ . To this end, we rewrite the numerator in (9) as

<!-- formula-not-decoded -->

where θ t,k is the angle between the two vectors x t,k -x ∗ ,k and E t [ d t,k ] . We absorb the quantities related to x ∗ ,k into a tunable parameter α t,k ≈ ‖ x t,k -x ∗ ,k ‖ cos θ t,k , which gives the stepsizes

<!-- formula-not-decoded -->

We emphasize that any α t,k we choose in practice may only be a (very rough) approximation of ‖ x t,k -x ∗ ,k ‖ cos θ t,k . In particular, while the optimal stepsizes ̂ γ t,k can be positive or negative, in practice it is very hard to estimate the sign of the inner product 〈 x t,k -x ∗ ,k , E t [ d t,k ] 〉 . Instead, we take the pragmatic approach of restricting α t,k &gt; 0 , effectively being optimistic that the expected search directions -E t [ d t,k ] always point towards x ∗ ,k for all k = 1 , . . . , m .

A further simplification is to use a common stepsize schedule α t across all blocks. This is often a reasonable choice for deep learning, where the model parameters are initialized randomly coordinatewise such that E [ ‖ x 0 ,k ‖ ] is constant for each coordinate k [e.g., 13, 19]. This brings us to

<!-- formula-not-decoded -->

We note that with some abuse of notation, here α t denotes a scalar, not a vector of ( α t, 1 , . . . , α t,k ) . This simplification reveals the connection between α t and the distance ‖ x t -x ∗ ‖ . Therefore, we expect α t to decrease as ‖ x t -x ∗ ‖ gradually shrinks. A simple strategy is to use a monotonic stepsize schedule on α t , such as the popular cosine decay [30] or linear decay [8].

Next, we need to replace the conditional expectations E t [ d t,k ] and E t [ ‖ d t,k ‖ 2 ] in (11) with computable approximations. We adopt the conventional approach of exponential moving average (EMA):

<!-- formula-not-decoded -->

where β ∈ [0 , 1) is the smoothing factor. This leads to a set of practical stepsizes:

<!-- formula-not-decoded -->

where we added a small constant glyph[epsilon1] &gt; 0 in the denominator to improve numerical stability.

## Algorithm 1 BCOS-g

<!-- formula-not-decoded -->

(same as RMSprop [49])

## Algorithm 2 BCOS-m

<!-- formula-not-decoded -->

## 2.3 Further simplification with one EMA estimator 106

107

108

109

The BCOS stepsizes in (13) are computed through the ratio of two online estimators ‖ u t,k ‖ and v t,k , which are susceptible to large variations because the numerator and denominator may fluctuate in different directions. In this section, we derive a simplified stepsize rule that depends only on v t,k .

110

111

112

First, recall the mean-variance decomposition of the conditional second moment,

<!-- formula-not-decoded -->

We interpret ‖ E t [ d t,k ] ‖ 2 as the signal power and Var t ( d t,k ) as the noise power, and define the signal fraction (SiF) as

<!-- formula-not-decoded -->

Apparently we have ρ t,k ∈ [0 , 1] . Using SiF, we can decompose the stepsizes in (10) as 113

<!-- formula-not-decoded -->

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

Now we can merge √ ρ t,k ∈ [0 , 1] into the tunable parameters α t,k and let α ′ t,k := α t,k √ ρ t,k . Then, following the same arguments as in Section 2.2, we arrive at the following simplified stepsize rule:

<!-- formula-not-decoded -->

where α ′ t is a scalar stepsize schedule, and v t,k is given in (12). The similarity between Adam and BCOS in (16) is apparent, and we will explain their connection in detail in the next section.

## 3 Instantiations of BCOS

The derivation of BCOS in Section 2 is carried out with a general search direction d t . In this section, we instantiate BCOS with two common choices of the search direction: the stochastic gradient and its EMA, also known as the stochastic momentum .

To simplify presentation, we focus on the case of single coordinate blocks , i.e., m = n and I k = { k } for k = 1 , . . . , n . Then we can express the EMA estimators for E t [ d 2 t,k ] in a vector form:

<!-- formula-not-decoded -->

where d 2 t denotes the element-wise squared vector d t glyph[circledot] d t . We also have s t = γ t ∈ R n and therefore

<!-- formula-not-decoded -->

where the vector of coordinate-wise stepsizes, γ t , can be expressed as

<!-- formula-not-decoded -->

Here √ v t denotes element-wise square roots, √ v t + glyph[epsilon1] means element-wise addition of glyph[epsilon1] , and the 126 fraction represent element-wise division or reciprocal. Again, the stepsize schedule α t is a scalar. We 127 no longer distinguish between α t and α ′ t because they are both tunable hyper-parameters. 128

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

161

162

163

## Algorithm 3 BCOS-c

<!-- formula-not-decoded -->

## 3.1 BCOS with EMA estimators

BCOS-g Algorithm 1 is the instantiation of BCOS using ∇ f ( x t , ξ t ) as the search direction. We call it BCOS-g to signify the use of gradient as search direction. The vector v t consists of coordinate-wise EMA estimators for E [ g 2 t,k ] , and the notation m t √ v t + glyph[epsilon1] means element-wise division.

We immediately recognize that BCOS-g is exactly the RMSprop algorithm [49], which is one of the first effective algorithms to train deep learning models. Our BCOS framework gives a novel interpretation of RMSprop and its effectiveness. In the special case with β = 0 and glyph[epsilon1] = 0 , we have v t = g 2 t , and both BCOS-g becomes the sign gradient method x t +1 = x t -α t sign ( g t ) , which also received significant attention in the literature [35, 2, 45, 23].

BCOS-m Using the stochastic momentum as search direction has a long history in stochastic approximation [e.g., 18, 34, 40]. It has become the default option for modern deep learning due to its superior performance compared with using plain stochastic gradients. Following the standard notation in machine learning, we use m t to denote the momentum, as shown in Algorithm 2. We call it BCOS-m to signify the use of momentum as the search direction. BCOS-m employs a second smoothing factor β 2 to calculate the EMA of m 2 t . These two smoothing factors β 1 and β 2 do not need to be the same and can be chosen independently in practice.

We notice that BCOS-m is very similar to Adam as given in (5) and (6). The difference is that in Adam, v t is the EMA of g 2 t instead of m 2 t . From BCOS perspective, Adam has a mismatch between the search direction m t and the second moment estimator based on g 2 t , which must be compensated for by a larger smoothing factor β 2 (because m t itself is a smoothed version of g t ). For BCOS-m, using β 2 = β 1 produces as good performance as Adam with the best tuned β 2 (see Section 5).

## 3.2 BCOS with conditional estimators

Recall that the optimal stepsizes ̂ γ t,k in (9) and their simplifications ˜ γ t,k in (11) and (15) are all based on conditional expectation. In Section 3.1, we used coordinate-wise EMA of d 2 t to approximate the conditional expectation E t [ d 2 t ] , i.e., v t as estimator of E t [ d 2 t ] in BCOS-g and of E t [ m 2 t ] in BCOS-m, respectively. In this section, we show that with m t as the search direction, we can exploit its update form to derive effective conditional estimators that can avoid using EMA.

We first repeat the definition of momentum here: m t = βm t -1 + (1 -β ) g t with β ∈ [0 , 1) . To derive an estimator of E t [ m 2 t ] , we expand the square and take expectation of each term:

<!-- formula-not-decoded -->

where we used the fact E t [ m 2 t -1 ] = m 2 t -1 and E t [ m t -1 ] = m t -1 thanks to the definition of E t [ · ] in (7). It remains to approximate E t [ g t ] and E t [ g 2 t ] . Clearly a good estimator for E t [ g t ] is m t . To approximate E t [ g 2 t ] , we could use a separate EMA estimator v ′ t = β ′ v ′ t -1 +(1 -β ′ ) g 2 t , but this introduces another algorithmic state v ′ t and a second smoothing factor β ′ . Meanwhile, we notice that the factor (1 -β ) 2 multiplying E t [ g 2 t ] is usually very small, especially for β close to 1. As a result, any error in approximating E t [ g 2 t ] is attenuated by a very small factor, so it may not cause much

## Algorithm 4 BCOSW-c

<!-- formula-not-decoded -->

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

Figure 1: Comparing AdamW and BCOSW-c with different momentum parameters.

<!-- image -->

difference. Therefore, for simplicity, we choose to approximate E t [ g 2 t ] with g 2 t itself. Combining with approximating E t [ g t ] with m t , we arrive at the following conditional estimator for E t [ m 2 t ] :

<!-- formula-not-decoded -->

While this can be a very effective estimator, we derive another one that is much simpler and as effective. The key is to approximate E [ g t ] in (19) with m t -1 instead of m t , which results in

<!-- formula-not-decoded -->

It resembles the standard EMA estimator in Adam, shown in (6), with an effective smoothing factor

<!-- formula-not-decoded -->

but with v t -1 replaced by m 2 t -1 . As a result, the estimator in (21) does not need to store v t -1 , thus requiring fewer optimizer states. This also explains that the second smoothing factor in Adam, β 2 , corresponding to β ′ here, should be much larger or closer to 1 than β . Specifically, β = 0 . 9 roughly corresponds to β ′ = 0 . 99 . The estimator in (21) eliminates β 2 as a second hyper-parameter.

Finally, replacing v t in BCOS-m with the one in (21) produces Algorithm 3. We call it BCOS-c to signify the conditional estimator. It has fewer optimizer states and fewer hyper-parameters to tune.

## 3.3 BCOS with decoupled weight decay

Weight decay is a common practice in training deep learning models to obtain better generalization performance. It can be understood as adding an L 2 regularization to the loss function, i.e., minimizing the regularized loss E ξ [ f ( x, ξ )] + λ 2 ‖ x ‖ 2 . Effectively, the stochastic gradient at x t becomes ∇ f ( x t , ξ t )+ λx t . We can apply the BCOS family of algorithms by simply replacing g t = ∇ f ( x t , ξ t ) with g t = ∇ f ( x t , ξ t ) + λx t . But a more effective way is to use decoupled weight decay as proposed in the AdamW algorithm [31]. Specifically, we apply weight decay separately in the BCOS update:

<!-- formula-not-decoded -->

We call the resulting method BCOSW following the naming convention of AdamW. Algorithm 4 shows BCOSW with the conditional estimator. Other variants (-g and -m) can be obtained similarly. A PyTorch implementation of all BCOS and BCOSW variants is given in Appendix A.

## 4 Convergence analysis

In this section, we present the convergence analysis of BCOS and BCOSW. Due to space limit, we focus on BCOSW and give comments on BCOS wherever apply. Our analysis consists of two stages. First, we analyze the convergence properties of the conceptual BCOSW method

<!-- formula-not-decoded -->

It is called 'conceptual' because we cannot compute E t [ d 2 t ] exactly in practice. Then, for the practical 188 BCOSW algorithm with stepsize γ t in (18), we bound the difference between the expected steps 189 E t [ γ t glyph[circledot] d t ] and E t [ ˜ γ t glyph[circledot] d t ] = ˜ γ t glyph[circledot] E t [ d t ] , which produces the desired convergence guarantee. 190

Figure 2: Comparing AdamW and BCOSW. Left: first 10k iterations; Right: all 100k iterations.

<!-- image -->

First, we need an appropriate condition to build our analysis. For the algorithm x t +1 = x t -˜ γ t glyph[circledot] d t , 191 the next iterate x t +1 moves closer to x ∗ in expectation if the expected direction -E t [ ˜ γ t glyph[circledot] d t ] aims 192 towards x ∗ and α t (a scalar) is sufficiently small. For the conceptual BCOS method, we have 193

<!-- formula-not-decoded -->

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

211

212

213

214

where sign ( · ) denotes element-wise sign function. Recall the definition of SiF in (14). With single coordinate blocks, we can write the vector of coordinate-wise SiFs as ρ t = E t [ d t ] 2 E t [ d 2 t ] ∈ [0 , 1] n . Then we have the expected update direction E t [ ˜ γ t glyph[circledot] d t ] = α t √ ρ t glyph[circledot] sign ( E t [ d t ]) . Since α t &gt; 0 is a scalar, we omit it from the statement of the aiming condition below.

Assumption A (Aiming condition) . There exists x ∗ ∈ R n such that

<!-- formula-not-decoded -->

holds for all t ≥ 0 almost surely. If d t is independent of the past trajectory conditioned on x t , i.e., E t [ d t ] = E [ d t | x t ] , then it suffices to have (23) hold for every x ∈ R n (independent of the trajectory).

Notice that we have E t [ d t ] = E [ d t | x t ] when, e.g., d t = ∇ f ( x t , ξ t ) and ξ t is independent of x t . The aiming conditions assume neither convexity nor smoothness, but it has some overlapping characteristics with convexity, which we discuss in Appendix B.

## 4.1 Analysis of conceptual BCOSW

Our first result concerns the one-step contraction property of the conceptual algorithm in (22).

Lemma 4.1. Suppose Assumption A holds, α t ≥ 0 and α t λ &lt; 1 . Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In fact, we can prove the following much stronger result of almost sure (a.s.) convergence.

Theorem 4.1. Suppose the stepsize schedule { α t } t ≥ 0 and weight decay parameter λ satisfy

<!-- formula-not-decoded -->

Then Assumption A implies ‖ x t -x ∗ ‖ → 0 a.s. for the conceptual BCOSW method (22) .

In terms of convergence rate, we can readily obtain linear convergence to a neighborhood of x ∗ with a constant α t based on (24). In addition, we have the following result on sublinear convergence.

Theorem 4.2. Consider the conceptual BCOSW method (22) with the stepsize schedule α t = α t +1 where 1 / 2 &lt; αλ &lt; 1 is satisfied. Then Assumption A implies that for all t ≥ 1 ,

<!-- formula-not-decoded -->

glyph[negationslash]

Without decoupled weight decay, BCOS may also have almost-sure convergence if the aiming 215 condition with λ = 0 holds with strict inequality for x t = x ∗ . However, the O (1 /t ) convergence rate 216 no longer holds. The proofs of the above results are given in Appendix C. 217

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

Figure 3: Left: Adam/ AdamW with β 1 , 2 = (0 . 9 , 0 . 99) . Right: BCOS/BCOSW with β = 0 . 9 .

<!-- image -->

## 4.2 Analysis of practical BCOSW

Now we consider the practical BCOSW method x t +1 = (1 -α t λ ) x t -γ t glyph[circledot] d t with the stepsize vector γ t given in (18). Our analysis is based on bounding the difference between the expected practical update E t [ γ t glyph[circledot] d t ] and the expected conceptual update E t [ ˜ γ t glyph[circledot] d t ] . Intuitively, it boils down to the quality of the estimator v t . Specifically, we need the following assumption on its bias.

Assumption B. There exists τ &gt; 0 and glyph[epsilon1] &gt; 0 such that for all t ≥ 0 it holds that

<!-- formula-not-decoded -->

Based on this assumption, we have the following bound on the expected update directions.

Lemma 4.2. Under Assumptions B, we have the following bound at each iteration t :

<!-- formula-not-decoded -->

where O ( Var t ( v t )) includes terms such as E t [( d t -E t [ d t ])( v t -E t [ v t ]) 2 ] and E t [( v t -E t [ v t ]) 3 ] and higher-order terms. The coefficient c t is defined as

<!-- formula-not-decoded -->

Here, SNR t ( · ) denotes conditional Signal-to-Noise Ratio . Specifically, SNR t ( d t ) = E t [ d t ] 2 Var t ( d t ) = ρ t 1 -ρ t and SNR t ( v t + glyph[epsilon1] ) = E [ v t + glyph[epsilon1] ] 2 Var t ( v t + glyph[epsilon1] ) = E [ v t + glyph[epsilon1] ] 2 Var t ( v t ) . This leads to the following result for practical BCOSW:

Theorem 4.3. Suppose Assumptions A and B holds, { α t } satisfies (25) and ‖ d t ‖ is bounded almost surely. Let δ be the smallest constant such that, for all t ≥ 0 ,

<!-- formula-not-decoded -->

Then we have lim sup t →∞ ‖ x t -x ∗ ‖ 2 ≤ δ 2 , meaning a.s. convergence to a neighborhood of x ∗ .

In fact, it is sufficient for λδ to be the lim sup →∞ of the left-hand side of (29) (see Appendix D.2). We notice from (28) that c t is small if the estimator v t has low bias (small τ ) and low variance (high SNR). In addition, it also helps to have high SNR of d t , for example, by using m t rather than g t .

Let's examine the bias-variance trade-off of the effective estimator v t used by popular optimizers:

- The classical SGD method (with d t = g t or d t = m t ) effectively uses a constant v t , which has zero variance but high bias | E t [ v t ] -E t [ d 2 t ] | = | v -E t [ d 2 t ] | for some constant v .
- Sign-SGD effectively uses v t = d 2 t , which has no bias but high variance Var t ( v t ) = Var t ( d t ) .
- The conditional estimator of BCOS-c has the following bias and variance (see Appendix E)

<!-- formula-not-decoded -->

Its bias is a small fraction of the bias of m t -1 and it has a very small variance.

- For Adam, we do not have a simple expression for its bias, but Var t ( v t ) = (1 -β 2 ) 2 Var t ( m 2 t ) .

In summary, our convergence analysis can be applied to a variety of different optimizers, including Adam and AdamW, by characterizing their bias-variance trade-off (see Appendix E).

245

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

281

282

Figure 4: Left: ResNet-20 on CIFAR10. Right: Vision Transformer on ImageNet.

<!-- image -->

## 5 Numerical experiments

We present preliminary experiments to compare BCOS with Adam, specifically their variants with decoupled weight decay. Among the BCOSW family, we focus on BCOSW-c (Algorithm 4).

Our first set of experiments are conducted on training the small GPT2 model with 124 million parameters [36] on the OpenWebText dataset [16]. We use global batch size 512 and run all experiments for 100k iterations with the first 2k for linear warmup and then cosine decay on { α t } . The default hyper-parameters are chosen (based on a coarse sweep) as: peak stepsize α max = 0 . 002 , final stepsize α min = 0 . 01 α max, glyph[epsilon1] = 10 -6 and weight decay λ = 0 . 1 .

Figure 1 (left) shows the test loss of AdamW with different combination of β 1 and β 2 . For each value of β 1 ∈ { 0 . 8 , 0 . 9 , 0 . 95 } , we choose the best β 2 after sweeping β 2 ∈ { 0 . 8 , 0 . 9 , 0 . 95 , 0 . 975 , 0 . 99 } . Their final loss achieved are all very close around 2 . 82 . For most ( β 1 , β 2 ) combinations, we observe loss spikes, especially at the beginning of the training (as shown in the inset). In contrast, Figure 1 (right) shows that BCOSW-c obtains the same final loss but with very smooth loss curve.

Figure 2 compares the test loss of AdamW against the three variants BCOSW-g, -m, and -c. We observe that BCOSW-g is significantly worse than the momentum-based methods. The loss curves for the momentum-based methods are all very close, but with spikes for both AdamW and BCOSW-m.

Figure 3 illustrates the difference between algoritms with and without decoupled weight decay. BCOS-c converges to much higher loss than BCOSW-c, and different values of λ (weight decay) makes dramatic difference for BCOS-c but cause little change to BCOSW-c. The same phenomenon happens for Adam versus AdamW, and we again observe spikes from their loss curve.

Finally, in Figure 4, we compare different algorithms for training ResNet-20 [20] on the CIFAR10 dataset [27], and also training the Vision Transformer (ViT) [50] on the ImageNet dataset [11]. For the ResNet task, we tried both cosine decay (drop by factor 100) and step decay (drop by 10 at epochs 80, 120, 150). The hyper-parameters chosen are: β = 0 . 9 for SGD and BCOSW-c, and β 1 , 2 = (0 . 9 , 0 . 99) for AdamW. We observe that the best-performing stepsize schedules are quite different for different methods. This prompt the need of tuning hyper-parameters for BCOSW for different tasks even though it shares similar tuned hyper-parameters as AdamW on the GPT2 task.

For the ViT task, although the best tuned stepsize schedules are similar between AdamW and BCOSW, their training and test curves look quite different. Figure 4 (right) shows that the test precision curves for BCOSW-c raises slowly but reaches slightly higher precision at the end.

These preliminary experiments demonstrate that BCOSW-c can obtain competitive performance compared with the state-of-the-art method AdamW, but with fewer optimizer states and fewer hyperparameters to tune. We are conducting additional empirical study to fully understand its potential.

## 6 Conclusion

BCOS is a stochastic approximation method that exploits the flexibility of taking different coordinatewise stepsizes. Rather than using sophisticated ideas from optimization such as preconditioning, it builds upon the simple idea of coordinate-wise contraction and focuses on constructing efficient statistical estimators, especially through conditional expectation, in determining the stepsizes.

283

## References

- [1] L. Balles and P. Hennig. Dissecting adam: The sign, magnitude and variance of stochastic 284 gradients. In International Conference on Machine Learning , pages 404-413. PMLR, 2018. 285
- [2] J. Bernstein, Y.-X. Wang, K. Azizzadenesheli, and A. Anandkumar. signsgd: Compressed 286 optimisation for non-convex problems. In International Conference on Machine Learning , 287 pages 560-569. PMLR, 2018. 288
- [3] J. R. Blum. Multidimensional Stochastic Approximation Methods. The Annals of Mathematical 289 Statistics , 25(4):737 - 744, 1954. 290
- [4] S. Bock, J. Goppold, and M. Weiß. An improvement of the convergence proof of the adam291 optimizer. arXiv preprint arXiv:1804.10587 , 2018. 292

293

294

295

- [5] X. Chen, C. Liang, D. Huang, E. Real, K. Wang, H. Pham, X. Dong, T. Luong, C.-J. Hsieh, Y. Lu, et al. Symbolic discovery of optimization algorithms. Advances in neural information processing systems , 36:49205-49233, 2023.
- [6] K. L. Chung. On a stochastic approximation method. The Annals of Mathematical Statistics , 296 pages 463-483, 1954. 297

298

299

- [7] K. L. Chung. On a stochastic approximation method. The Annals of Mathematical Statistics , pages 463-483, 1954.

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

320

321

322

323

324

325

326

- [8] A. Defazio, A. Cutkosky, H. Mehta, and K. Mishchenko. Optimal linear decay learning rate schedules and further refinements, 2024.
- [9] A. Défossez, L. Bottou, F. Bach, and N. Usunier. A simple convergence proof of adam and adagrad. arXiv preprint arXiv:2003.02395 , 2020.
- [10] B. Delyon and A. Juditsky. Accelerated stochastic approximation. SIAM Journal on Optimization , 3(4):868-881, 1993.
- [11] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition , pages 248-255, 2009.
- [12] C. Derman and J. Sacks. On Dvoretzky's Stochastic Approximation Theorem. The Annals of Mathematical Statistics , 30(2):601 - 606, 1959.
- [13] E. Dinan, S. Yaida, and S. Zhang. Effective theory of transformers at initialization. arXiv preprint arXiv:2304.02034 , 2023.
- [14] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research , 12(Jul):2121-2159, 2011.
- [15] A. Dvoretzky. On stochastic approximation. In Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability , volume 1, pages 39-55. University of California Press, 1956.
- [16] A. Gokaslan and V. Cohen. Openwebtext corpus. http://Skylion007.github.io/ OpenWebTextCorpus , 2019.
- [17] D. M. Gomes, Y. Zhang, E. Belilovsky, G. Wolf, and M. S. Hosseini. Adafisher: Adaptive second order optimization via fisher information. arXiv preprint arXiv:2405.16397 , 2024.
- [18] A. M. Gupal and L. T. Bazhenov. A stochastic analog of the conjugate gradient method. Cybernetics , 8(1):138-140, 1972.
- [19] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision , pages 1026-1034, 2015.
- [20] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In 2016 327 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 770-778, 2016. 328

- [21] D. Hwang. Fadam: Adam is a natural gradient optimizer using diagonal empirical fisher 329 information. arXiv preprint arXiv:2405.12807 , 2024. 330

331

332

- [22] R. A. Jacobs. Increased rates of convergence through learning rate adaption. Neural Networks , 1:295-307, 1988.

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

369

370

371

372

373

374

- [23] W. Jiang, S. Yang, W. Yang, and L. Zhang. Efficient sign-based optimization: Accelerating convergence via variance reduction. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [24] K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, and J. Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024.
- [25] H. Kesten. Accelerated stochastic approximation. Annals of Mathematical Statistics , 29(1):4159, 1958.
- [26] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In Proceedings of International Conference on Learning Representations (ICLR) , 2015. arXiv:1412.6980.
- [27] A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Technical Report 0, University of Toronto, Toronto, Ontario, 2009.
- [28] F. Kunstner, A. Milligan, R. Yadav, M. Schmidt, and A. Bietti. Heavy-tailed class imbalance and why adam outperforms gradient descent on language models. Advances in Neural Information Processing Systems , 37:30106-30148, 2024.
- [29] W. Lin, F. Dangel, R. Eschenhagen, J. Bae, R. E. Turner, and A. Makhzani. Can we remove the square-root in adaptive gradient methods? a second-order perspective. arXiv preprint arXiv:2402.03496 , 2024.
- [30] I. Loshchilov and F. Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 , 2016.
- [31] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR) , 2019.
- [32] A. R. Mahmood, R. S. Sutton, T. Degris, and P. M. Pilarski. Tuning-free step-size adaption. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 2121-2124, 2012.
- [33] F. Mirzoakhmedov and S. P. Uryasev. Adaptive step adjustment for a stochastic optimization algorithm. Zh. Vychisl. Mat. Mat. Fiz. , 23(6):1314-1325, 1983. [U.S.S.R. Comput. Math. Math. Phys. 23:6, 1983].
- [34] B. T. Polyak. Comparison of the rates of convergence of one-step and multi-step optimization algorithms in the presence of noise. Engineering Cybernetics , 15:6-10, 1977.
- [35] B. T. Polyak and Y. Z. Tsypkin. Pseudogradient adaptation and training algorithms. Automation and Remote Control, a translation of Avtomatika i Telemekhanika , 34(3):377-397, 1973.
- [36] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. OpenAI Tech Report, 2019.
- [37] S. J. Reddi, S. Kale, and S. Kumar. On the convergence of adam and beyond. arXiv preprint arXiv:1904.09237 , 2019.
- [38] H. Robbins and S. Monro. A stochastic approximation method. The Annals of Mathematical Statistics , 22(3):400-407, 1951.
- [39] H. Robbins and D. Siegmund. A convergence theorem for non negative almost supermartingales and some applications. In J. S. Rustagi, editor, Optimizing Methods in Statistics , pages 233-257. Academic Press, 1971.
- [40] A. Ruszczy´ nski and W. Syski. Stochastic approximation method with gradient averaging for unconstrained problems. IEEE Transactions on Automatic Control , 28(12):1097-1105, 1983.

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

- [41] A. Ruszczy´ nski and W. Syski. Stochastic approximation algorithm with gradient averaging and on-line stepsize rules. In J. Gertler and L. Keviczky, editors, Proceedings of 9th IFAC World Congress , pages 1023-1027, Budapest, Hungary, 1984.
- [42] A. Ruszczy´ nski and W. Syski. A method of aggregate stochastic subgradients with on-line stepsize rules for convex stochastic programming problems. Mathematical Programming Study , 28:113-131, 1986.
- [43] A. Ruszczy´ nski and W. Syski. On convergence of the stochastic subgradient method with on-line stepsize rules. Journal of Mathematical Analysis and Applications , 114:512-527, 1986.
- [44] J. Sacks. Asymptotic distribution of stochastic approximation procedures. The Annals of Mathematical Statistics , 29(2):373-405, 1958.
- [45] M. Safaryan and P. Richtarik. Stochastic sign descent methods: New algorithms and better theory. In M. Meila and T. Zhang, editors, Proceedings of the 38th International Conference on Machine Learning , volume 139 of Proceedings of Machine Learning Research , pages 9224-9234. PMLR, 18-24 Jul 2021.
- [46] N. N. Schraudolph. Local gain adaptation in stochastic gradient descent. In Proceedings of Nineth International Conference on Artificial Neural Networks (ICANN) , pages 569-574, 1999.
- [47] R. S. Sutton. Adapting bias by gradient descent: An incremental version of Delta-Bar-Delta. In Proceedings of the Tenth National Conference on Artificial Intelligence (AAAI'92) , pages 171-176. The MIT Press, 1992.
- [48] T. Tieleman. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning , 4(2):26, 2012.
- [49] T. Tieleman and G. Hinton. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning , 4(2):26-31, 2012.
- [50] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jegou. Training dataefficient image transformers and distillation through attention. In M. Meila and T. Zhang, editors, Proceedings of the 38th International Conference on Machine Learning , volume 139 of Proceedings of Machine Learning Research , pages 10347-10357. PMLR, 18-24 Jul 2021.
- [51] J. H. Venter. On Dvoretzky Stochastic Approximation Theorems. The Annals of Mathematical Statistics , 37(6):1534 - 1544, 1966.
- [52] M. T. Wasan. Stochastic Approximation . Cambridge University Press, 1969.
- [53] Z. Yao, A. Gholami, S. Shen, M. Mustafa, K. Keutzer, and M. Mahoney. Adahessian: An adaptive second order optimizer for machine learning. In proceedings of the AAAI conference on artificial intelligence , volume 35, pages 10665-10673, 2021.
- [54] Y. Zhang, C. Chen, Z. Li, T. Ding, C. Wu, D. P. Kingma, Y. Ye, Z.-Q. Luo, and R. Sun. Adam-mini: Use fewer learning rates to gain more. arXiv preprint arXiv:2406.16793 , 2024.
- [55] Y. Zhang, C. Chen, N. Shi, R. Sun, and Z.-Q. Luo. Adam can converge without any modification on update rules. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems , volume 35, pages 28386-28399. Curran Associates, Inc., 2022.
- [56] F. Zou, L. Shen, Z. Jie, W. Zhang, and W. Liu. A sufficient condition for convergences of adam and rmsprop. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition , pages 11127-11135, 2019.

## A PyTorch implementation of BCOS 417

```
import torch 418 from torch.optim import Optimizer 419 420 class BCOS_short(Optimizer): 421 def __init__(self, params , lr, beta=0.9, eps=1e-6, 422 weight_decay=0.1, mode='c', decouple_wd=True): 423 424 defaults = dict(lr=lr, beta=beta, eps=eps, wd=weight_decay) 425 super().__init__(params , defaults) 426 427 if mode not in ['g', 'm', 'c']: 428 raise ValueError(f"BCOS mode {mode} not supported") 429 self.mode = mode 430 self.decouple_wd = decouple_wd # True for BCOSW 431 432 def step(self, closure = None): 433 434 for group in self.param_groups: 435 lr = group["lr"] 436 beta = group["beta"] 437 eps = group["eps"] 438 wd = group["wd"] 439 440 for p in group["params"]: 441 if not p.requires_grad: 442 continue 443 444 state = self.state[p] 445 g = p.grad 446 447 # initialize optimizer states for specific modes 448 if self.mode in ['m', 'c'] and 'm' not in state: 449 state['m'] = g.detach().clone() 450 if self.mode in ['g', 'm'] and 'v' not in state: 451 state['v'] = g.detach().square() 452 453 # decoupled weight decay or absorb in gradient 454 if self.decouple_wd: # p := (1 -lr * wd) * p 455 p.data.mul_(1 -lr * wd) 456 else: # g := g + wd * p 457 g.data.add_(p.data, alpha = wd) 458 459 if self.mode in ['m', 'c']: 460 m = state['m'] 461 if self.mode == 'c': 462 beta_v = 1 -(1 -beta)**2 463 g2 = g.detach().square() 464 v = beta_v * m.square() + (1 -beta_v) * g2 465 # update momentum 466 m.mul_(beta).add_(g.detach(), alpha=1 -beta) 467 d = m 468 else: 469 d = g.detach() 470 471 if self.mode in ['g', 'm']: # EMA estimator 472 v = state['v'] 473 v.mul_(beta).add_(d.square(), alpha=1 -beta) 474 475 # BCOS update: p := p -lr * (d / (sqrt(v) + eps)) 476 p.data.add_(d.div(v.sqrt() + eps), alpha= -lr) 477
```

Listing 1: BCOS and BCOSW implementation as a single PyTorch Optimizer

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

506

507

## B Aiming condition and convexity

In the paper we have focused on the special case of single coordinate blocks. To investigate the relation between the aiming condition and convexity, it is more instructive to examine the general block structure. For general block partitions ∪ m k =1 I k , employing a block-coordinate stepsize vector s t where each block I k of s t is defined as s t,k = ˜ γ t,k 1 n k yields iterative methods of the form

<!-- formula-not-decoded -->

with conceptual BCOS stepsizes

<!-- formula-not-decoded -->

The corresponding aiming condition is as follows, which guarantees one-step improvement.

Assumption C. There exists x ∗ ∈ R n such that

<!-- formula-not-decoded -->

holds for all t ≥ 0 almost surely. If d t is independent of the past trajectory conditioned on x t , i.e., E t [ d t ] = E [ d t | x t ] , then it suffices to have (31) hold for every x ∈ R n .

Assumption C allows us to conduct a comparative analysis of the aiming condition and the classical convexity assumption, highlighting their similarities and key differences. For the sake of simplicity in our exposition, we will assume that the stochastic search direction d t is trajectory independent, i.e., E t [ d t ] = E [ d t | x t ] , allowing us to drop the subscript t . We further assume that d t satisfies E [ d ] = ∇ f ( x ) . Simplifying (31):

<!-- formula-not-decoded -->

In the specific case of a full-dimensional block stepsize, where ˜ γ t = 1 ‖∇ f ( x ) ‖ ∈ R + is a scaler and the stepsize vector is s t = ˜ γ t 1 n , the aiming condition simplifies to:

<!-- formula-not-decoded -->

Condition (33) is directly implied by the classical convex assumption, which states:

<!-- formula-not-decoded -->

To see the implication, simply substitute y = x ∗ and ∇ f ( x ∗ ) = 0 into the above convex inequality.

However, the aiming condition under a general block partition exhibits a significant departure from the classical notion of convexity, as expected update directions deviate from true gradients and become axis-aligned. Consider the extreme case of coordinate-wise stepsizes, where s t = ˜ γ t ∈ R n and each element is chosen as ˜ γ t,k = 1 √ ∇ f ( x k ) 2 = 1 |∇ f ( x k ) | . The specific choice of stepsize yields an aiming condition of the form:

<!-- formula-not-decoded -->

To illustrate the fundamental differences between this coordinate-wise aiming condition (35) and the standard convexity assumption (34), we provide the following two counterexamples, each satisfying one condition while failing the other:

- Aiming but not convex : Let f ( x ) := log( x ) with the optimal solution x ∗ = 0 . On the domain of R + , the gradient is f ′′ ( x ) = 1 x , and thus sign ( f ′ ( x )) = 1 for all x &gt; 0 . Consequently, for any x ∈ R + , we have

<!-- formula-not-decoded -->

satisfying the aiming condition (35). However, log( x ) is a concave function, thus failing the 508 convex inequality (34). 509

- Convex but not aiming : Consider the quadratic function class f : R 2 → R , f ( x ) = 1 2 x T Ax . Choose coefficient matrix A :

<!-- formula-not-decoded -->

Since A is positive semidefinite, the function f is convex and attains its minimum at x ∗ = 0 . The gradient of f is

<!-- formula-not-decoded -->

Evaluating the aiming condition (35) at x = (1 . 5 , 1) T , we get 510

<!-- formula-not-decoded -->

- Thus, the aiming condition (35) at this point even though f is convex. 511

## C Convergence analysis of conceptual BCOSW 512

First, we notice that the aiming condition is Assumption A is equivalent to 513

because 514

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We use it to prove Lemma 4.1. 515

516

<!-- formula-not-decoded -->

Under Assumption A, the aiming condition in (36) implies that the inner product in the last equality 517 above is non-negative. With α t ≥ 0 and α t λ ≤ 1 , we can drop the inner product term to obtain 518

<!-- formula-not-decoded -->

where the last inequality follows from the loose upper bound ρ t,k sign ( E t [ d t,k ]) ≤ 1 . 519

<!-- formula-not-decoded -->

The proof of Theorem 4.1 follows from the following almost supermartingale lemma. 520

LemmaC.1 ('Almost supermartingale", Theorem 1 [39]) . ) Let (Ω , F , P ) be a probability space, and 521 F 0 ⊂ F 1 ⊂ . . . be a sequence of subσ -algebras of F . For each t , let X t , a t , b t , c t be non-negative 522 F t -measurable random variables such that 523

<!-- formula-not-decoded -->

Given ∑ ∞ t =0 a t &lt; ∞ and ∑ ∞ t =0 b t &lt; ∞ , then lim t →∞ X t exists and is finite, and ∑ ∞ t =0 c t &lt; ∞ 524 almost surely (a.s.). 525

Proof of Theorem 4.1. Define X t := ‖ x t -x ∗ ‖ 2 and F t to be the σ -algebra generated by 526 X 0 , · · · , X t . Lemma 4.1 implies the following recursive relationship 527

<!-- formula-not-decoded -->

In the form of (37), we have a t = α 2 t λ 2 , b t = α 2 t c ∗ , c t = 2 α t λ ‖ x t -x ∗ ‖ 2 . Here, X t , a t , b t , c t are 528 trivially non-negative, and the squared summable assumption of α t guarantees: 529

<!-- formula-not-decoded -->

So far, we have verified all the assumptions in Lemma C.1, so we conclude that 530

<!-- formula-not-decoded -->

This is compatible with ∑ ∞ t =0 α t = ∞ only if 531

<!-- formula-not-decoded -->

as desired. 532

533

534

To quantify the convergence rate, we study the upper bound on the expected distance to the optimal solution E [ ‖ x T -x ∗ ‖ 2 ] , after recursively applying BCOSW for T iterations.

535

536

Theorem C.1. Suppose Assumption A holds, α t ≥ 0 and α t λ ≤ 1 . The expected distance to x ∗ admits the following upper bound after T iterations of BCOSW:

<!-- formula-not-decoded -->

where c ∗ := ( n + λ 2 ‖ x ∗ ‖ 2 +2 λ ‖ x ∗ ‖ 1 ) denote the constant residual that depends on x ∗ . 537

<!-- formula-not-decoded -->

Proof. Taking expectation of the recursive relationship (24) and applying the law of total expectation, 538 we obtain: 539

<!-- formula-not-decoded -->

540

541

542

543

544

545

546

where a &gt; p &gt; 0 , b &gt; 0 . Then 547

as desired.

Different choices of stepsize schedule lead to different convergence behaviors. Next, we consider two choices of α t : (i) diminishing learning rates α t = α t +1 , which leads to Theorem 4.2 and (ii) constant learning rates α t = α which lead to linear convergence to a neighborhood of x ∗ .

The proof of Theorem 4.2 is a direct application of a classical result in the 1954 paper of Chung's [7].

Lemma C.2 (Chung's lemma, Lemma 1 from [7]) . Suppose that { X t } is a sequence of real numbers such that for t ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Theorem 4.2. Taking expectation of both sides of (24) with α t = α t +1 at iteration T , we 548 have 549

<!-- formula-not-decoded -->

where the last inequality is in light of (38) in Theorem C.1 and α t = α t +1 . Upper bounding 550 ( 1 -αλ t +1 ) 2 by 1 yields: 551

<!-- formula-not-decoded -->

Further replacing the finite sum ∑ T -2 t =0 1 ( t +1) 2 = ∑ T -1 t =1 1 t 2 by its infinite version π 2 6 , we obtain a 552 recursive relationship in the form of (39): 553

<!-- formula-not-decoded -->

554

555

556

557

558

559

560

561

562

with X t = E [ ‖ x t -1 -x ∗ ‖ 2 ] , a = 2 αλ, b = α 2 c ∗ + α 2 λ 2 ( E [ ‖ x 0 -x ∗ ‖ 2 ] + π 2 α 2 c ∗ 6 ) , and p = 1 , which satisfies the Chung's assumptions a &gt; 1 = p &gt; 0 , b &gt; 0 because αλ ∈ (0 . 5 , 1) . Lemma C.2 implies

<!-- formula-not-decoded -->

as desired.

With a constant stepsize, we obtain linear convergence to a neighborhood of x ∗ , as stated in the following corollary.

Corollary C.2. Fix learning rate schedule α t = α where α satisfies αλ &lt; 1 . Let x t 's be a sequence generated by applying the conceptual BCOSW. Under Assumption A, the asymptotic expected distance to x ∗ admits the following upper bound:

<!-- formula-not-decoded -->

Proof. A direct application of Theorem C.1 with α t = α yields the following upper bound on: 563 E [ ‖ x T -x ∗ ‖ 2 ] 564

<!-- formula-not-decoded -->

which decreases exponentially with T 565

566

<!-- formula-not-decoded -->

and converges to a constant.

## D Convergence analysis of practical BCOSW

567

568

569

## D.1 Proof of Lemma 4.2

We first prove Lemma 4.2. To proceed, we decompose the error between the expected search directions into two parts (elementwise inequality between vectors):

<!-- formula-not-decoded -->

Under certain assumptions on the quality of the estimator v t , we demonstrate that the practical update 570 approximates the conceptual update in expectation by bounding the two terms on the right-hand side 571 separately. 572

Assumption B leads to an upper bound for the first error term in (41). 573

Lemma D.1. Under Assumption B, it holds that: 574

<!-- formula-not-decoded -->

Proof. The proof leverages the second-order Taylor expansion of g ( y ) := 1 √ y : 575

<!-- formula-not-decoded -->

Applying Taylor expansion at y := E t [ d 2 t ] with δ := E t [ v t ] + glyph[epsilon1] -E t [ d 2 t ] yields the following 576 approximation: 577

<!-- formula-not-decoded -->

578

579

580

581

582

where (43) is a consequence of Assumption B.

To establish the upper bound on the second error term in (41), ∣ ∣ ∣ ∣ E t [ d t ] √ E t [ v t ]+ glyph[epsilon1] -E t [ d t √ v t + glyph[epsilon1] ] ∣ ∣ ∣ ∣ , we present a useful approximation for general differential function g .

Lemma D.2. For any differentiable function g and random variable X ∈ R n , the following expansion holds:

<!-- formula-not-decoded -->

where 〈· , ·〉 denotes matrix inner product, i.e, 〈 A,B 〉 = Tr ( A B ) , and p ∈ N and

<!-- formula-not-decoded -->

Proof. Let δ := X -E [ X ] . The second-order Taylor expansion of g at E [ X ] yields 583

<!-- formula-not-decoded -->

Taking expectation with respect to X , we have 584

<!-- formula-not-decoded -->

where 585

The following lemma provides an approximation for E [ g ] with g ( Y, Z ) := Y √ Z . 586 Lemma D.3. Let Y, Z be two random variables and Z &gt; 0 almost surely, then 587

<!-- formula-not-decoded -->

Proof. We apply Lemma D.2 with X := ( Y, Z ) and g ( x ) = g ( y, z ) := y √ z . First, the gradient and 588 Hessian of g can be calculated as 589

<!-- formula-not-decoded -->

For general p -th partial derivative, we derive the following result for any q ∈ [0 , p ] : 590

<!-- formula-not-decoded -->

which Substitute the gradient, Hessian and p -th order partial derivative into (44), we get 591

<!-- formula-not-decoded -->

as desired. 592

A combination of the consequence of Lemma D.3 and Assumption B culminates in an upper bound 593 on the second error term. 594

Lemma D.4. Define signal-noise-ratio SNR t ( Y ) := E t [ Y t ] 2 Var t ( Y t ) . Under Assumptions B, we have 595

<!-- formula-not-decoded -->

Proof. Following Lemma D.3 with Y := d t , Z := v t + glyph[epsilon1] , we get 596

<!-- formula-not-decoded -->

We express the covariance between d and v + glyph[epsilon1] based on the definitions of SNR as follows: 597

<!-- formula-not-decoded -->

where SNR t ( d t ) is closely connected to the signal fraction ρ t , defined as ρ t := E t [ d t ] 2 E t [ d 2 t ] :

<!-- formula-not-decoded -->

The first term (46) admits the following upper bound: 598

<!-- formula-not-decoded -->

where (47) is given by Lemma D.1. 599

Combining the upper bounds on two terms on the right-hand side of (41), we finally can prove 600 Lemma 4.2 601

Proof of Lemma 4.2. It follows immediately by triangle inequality, Lemma D.1 and Lemma D.4. 602

<!-- formula-not-decoded -->

Finally, bounding | Corr t ( d t , v t + glyph[epsilon1] ) | by 1 and recognizing 1 ρ t -1 = 1 SNR t ( d t ) give the desired 603 result. 604

605

606

607

608

609

610

611

## D.2 Proof of Theorem 4.3

To prove Theorem 4.3, we can use a classical result on stochastic approximation originally due to Dvoretzky [15].

Theorem D.1 (An extension of Dvoretzky's Theorem) . Let (Ω = { ω } , F , P ) be a probability space. Let { x t } and { y t } be sequences of random variables such that, for all t ≥ 0 ,

<!-- formula-not-decoded -->

where the transformation T t satisfy, for any x 0 , . . . , x t ∈ R n ,

<!-- formula-not-decoded -->

and the sequences { a t } , { b t } , { c t } and { d t } are non-negative and satisfy

<!-- formula-not-decoded -->

In addition, suppose the following conditions hold with probability one: 612

<!-- formula-not-decoded -->

Then we have with probability one, 613

<!-- formula-not-decoded -->

614

615

616

617

618

619

620

621

622

623

624

625

Remark. There are many extensions of Dvoretzky's original results [15]. Theorem D.1 is a minor variation of Venter [51, Theorem 1]. More concretely,

- Theorem 1 of Venter [51] has the sequence { a t } being a constant sequence, i.e., a t = a ∞ for all t ≥ 0 . The extension to a non-constant sequence { a t } is outlined in the original work of Dvoretzky [15] and admits a simple proof due to Derman and Sacks [12].
- Theorem 1 of Venter [51] does not include the sequence { d t } . The extension with ∑ ∞ t =0 d t &lt; ∞ is straightforward based on a simple argument of Dvoretzky [15].
- More generally, the sequences { a t } , { b t } , { c t } , { d t } can be non-negative measurable functions of x 0 , . . . , x t , and the conclusion of Theorem D.1 holds if a ∞ is an upper bound on lim sup t →∞ a t ( x 0 , . . . , x t ) uniformly for all sequences x 0 , . . . , x t , . . . [12, 39].

We also need the following lemma.

Lemma D.5. Under Assumptions B, it holds that

<!-- formula-not-decoded -->

where c t is given by (28) . 626

Proof. Adding and subtracting the term E t [ d t ] √ E t [ d 2 t ] from the inner product, we obtain: 627

<!-- formula-not-decoded -->

where the inequality (51) is due to Lemma 4.2. To finish the proof, we recall √ ρ t = E t [ d t ] √ E t [ d 2 t ] . 628

Proof of Theorem 4.3. We can write the practical BCOSW algorithm as 629

<!-- formula-not-decoded -->

In terms of the decomposition in (48), we have x t +1 = T t ( x 0 , . . . , x t ) + y t where 630

<!-- formula-not-decoded -->

Apparently we have E t [ y t ] = E [ y t | x 0 , . . . , x t ] = 0 . We also have ∑ ∞ t =0 E [ ‖ y t ‖ 2 ] &lt; ∞ with a 631 bounded assumption on y t due to the assumption ∑ ∞ t =0 α 2 t &lt; ∞ . 632

The squared distance between T t ( x 0 , . . . , x t ) and x ∗ is 633

<!-- formula-not-decoded -->

From Lemma D.5 and the aiming condition (36), we have 634

<!-- formula-not-decoded -->

In addition, by the bounded assumption on d t , there exist a constant B such that

<!-- formula-not-decoded -->

Together with 0 &lt; 1 -α t λ &lt; 1 , we conclude that 635

<!-- formula-not-decoded -->

We observe that there exist δ &gt; 0 such that 636

<!-- formula-not-decoded -->

637

638

639

640

641

642

Therefore, ‖ x t -x ∗ ‖ ≥ δ implies

<!-- formula-not-decoded -->

Otherwise, when ‖ x t -x ∗ ‖ ≤ δ , we have

<!-- formula-not-decoded -->

By defining a t as the right-hand side of the above inequality, i.e.,

<!-- formula-not-decoded -->

we can combine the above two cases as

<!-- formula-not-decoded -->

With the additional definition of

<!-- formula-not-decoded -->

we arrive at the key inequality (49).

We are left to check the conditions in (50). Using the assumptions on { α t } , the definition in (52) 643 implies that a t converges and lim t →∞ a t = δ 2 . The conditions on { b t } and { d t } are automatically 644 satisfied. For { c t } , if ∑ ∞ t =0 c t &lt; ∞ , then we must have ‖ x t -x glyph[star] ‖ 2 → 0 almost surely and the 645 conclusion of theorem holds trivially. Otherwise, ∑ ∞ t =0 c t = ∞ allows all the conditions in (50) to 646 hold, so we can invole Theorem D.1 to conclude the proof. 647

## E Biases and variances of second-moment estimators 648

649

Lemma 4.2 provides guidelines for choosing the second-moment estimator v t , which should exhibit:

650

651

652

653

- low bias (i.e., low τ );
- high signal-to-noise ratio (i.e., high SNR );

However, there is always a bias-variance tradeoff for various estimators v t . Here are some examples:

1. Sign-SGD is equivalent to take v t = d 2 t , exhibiting low bias and high variance

<!-- formula-not-decoded -->

with resulting update rule to be sign-SGD (with and without momentum corresponding to 654 d t = g t and d t = m t respectively): 655

<!-- formula-not-decoded -->

2. Standard SGD is equivalent to take v t = c for some positive constant c , exhibiting high bias 656 and low variance 657

<!-- formula-not-decoded -->

with resulting update rule to be SGD (with and without momentum corresponding to d t = g t 658 and d t = m t respectively): 659

<!-- formula-not-decoded -->

where α ′ t := α t √ c + glyph[epsilon1] . 660

3. BCOS-m uses v t = EMA β ( d 2 t ) , exhibiting non-trivial bias and low variance properties: 661

<!-- formula-not-decoded -->

As for the variance, we get 662

<!-- formula-not-decoded -->

4. Adam uses estimator v t = EMA β 2 ( g 2 t ) with search direction d t = EMA β 1 ( g t ) : exhibiting 663 non-trivial bias and low variance properties: 664

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As for the variance, we get 665

<!-- formula-not-decoded -->

5. BCOS-c uses estimator v t = (1 -(1 -β ) 2 ) m 2 t -1 +(1 -β ) 2 g 2 t with search direction d t = 666 m t = EMA β ( g t ) = βm t -1 +(1 -β ) g t : exhibiting low bias and low variance properties: 667

<!-- formula-not-decoded -->

668

<!-- formula-not-decoded -->

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

686

687

688

689

690

691

692

693

694

695

696

697

698

699

700

701

702

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

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims match theoretical and experimental results presented.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the need of tuning the stepsize schedule in Sections 2.2, 2.3 and 5.

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

Justification: We state the assumptions and key results clearly and give rigorous proofs.

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

755

756

757

758

759

760

761

762

763

764

765

766

767

768

769

770

771

772

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

Justification: We describe the hyper-parameters used in the experiments clearly, and included an optimizer implementation in Appendix A that is used to generate the experiment results.

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

Justification: The data used in this paper are all widely available in the public domain. We also include the optimizer code in Appendix A for reproducibility.

773

774

775

776

777

778

779

780

781

782

783

784

785

786

787

788

789

790

791

792

793

794

795

796

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

812

813

814

815

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

Justification: We describe the hyper-parameters used in the Experiment section. The models and datasets we use are all very standard and there should be no confusion on the settings.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Performing the errors bars are computationally costly and they are not essential in understanding and justifying the results in this paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

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

840

841

842

843

844

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

871

872

873

874

875

876

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No]

Justification: Information on compute resources are not relevant to the results of this paper. We focus on training performance of standard tasks whose compute requirements are well-known.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We reviewed the Code of Ethics and stick with it.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no perceivable negative impact of the work perfomed.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

877

878

879

880

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

892

893

894

895

896

897

898

899

900

901

902

903

904

905

906

907

908

909

910

911

912

913

914

915

916

917

918

919

920

921

922

923

924

925

926

927

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: This paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All sources are properly cited and no license required.

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

Justification: This paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.

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

- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: the core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
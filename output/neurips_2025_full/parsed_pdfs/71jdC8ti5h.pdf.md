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

## Randomness Helps Rigor: A Probabilistic Learning Rate Scheduler Bridging Theory and Deep Learning Practice

## Anonymous Author(s)

Affiliation Address email

## Abstract

Learning rate schedulers have shown great success in speeding up the convergence of learning algorithms in practice. However, their convergence to a minimum has not been proven theoretically. This difficulty mainly arises from the fact that, while traditional convergence analysis prescribes to monotonically decreasing (or constant) learning rates, schedulers opt for rates that often increase and decrease through the training epochs. In this work, we aim to bridge the gap by proposing a probabilistic learning rate scheduler (PLRS) that does not conform to the monotonically decreasing condition, with provable convergence guarantees. To cement the relevance and utility of our work in modern day applications, we show experimental results on deep neural network architectures such as ResNet, WRN, VGG, and DenseNet on CIFAR-10, CIFAR-100, and Tiny ImageNet datasets. We show that PLRS performs as well as or better than existing state-of-the-art learning rate schedulers in terms of convergence as well as accuracy. For example, while training ResNet-110 on the CIFAR-100 dataset, we outperform the state-of-the-art knee scheduler by 1 . 56% in terms of classification accuracy. Furthermore, on the Tiny ImageNet dataset using ResNet-50 architecture, we show a significantly more stable convergence than the cosine scheduler and a better classification accuracy than the existing schedulers.

## 1 Introduction

Over the last two decades, there has been an increased interest in analyzing the convergence of gradient descent-based algorithms. This can be majorly attributed to their extensive use in the training of neural networks and their numerous derivatives. Stochastic Gradient Descent (SGD) and their adaptive variants such as Adagrad [8], Adadelta [31], and Adam [17] have been the choice of optimization algorithms for most machine learning practitioners, primarily due to their ability to process enormous amounts of data in batches. Even with the introduction of adaptive optimization techniques that use a default learning rate, the use of stochastic gradient descent with a tuned learning rate was quite prevalent, mainly due to its generalization properties [34]. However, tuning the learning rate of the network can be computationally intensive and time consuming.

- Various methods to efficiently choose the learning rate without excessive tuning have been explored. 29
- One of the initial successes in this domain is the random search method [3]; here, a learning rate is 30
- randomly selected from a specified interval across multiple trials, and the best performing learning 31
- rate is ultimately chosen. Following this, more advanced methods such as Sequential Model-Based 32
- Optimization (SMBO) [4] for the choice of learning rate became prevalent in practice. SMBO 33
- represents a significant advancement over random search by tracking the effectiveness of learning 34
- rates from previous trials and using this information to build a model that suggests the next optimal 35

learning rate. A tuning method for shallow neural networks based on theoretical computation of the 36 Hessian Lipschitz constant was proposed by Tholeti et al. [27]. 37

Several works on training deep neural networks prescribed the use of a decaying Learning Rate (LR) 1 38 scheduler [10, 32, 26]. Recently, much attention has been paid to cyclically varying learning rates 39 [24]. By varying learning rates in a triangular schedule within a predetermined range of values, the 40 authors hypothesize that the optimal learning rate lies within the chosen range, and the periodic 41 high learning rate helps escape saddle points. Although no theoretical backing has been provided, 42 it was shown to be a valid hypothesis owing to the presence of many saddle points in a typical 43 high dimensional learning task [6]. Many variants of the cyclic LR scheduler have henceforth been 44 used in various machine learning tasks [12, 7, 1]. A cosine-based cyclic LR scheduler proposed by 45 Loshchilov et al. [21] has also found several applications, including Transformers [30, 5]. Following 46 the success of the cyclic LR schedulers, a one-cycle LR scheduler proposed by Smith et al. [25] 47 has been observed to provide faster convergence empirically; this was attributed to the injection of 48 'good noise' by higher learning rates which helps in convergence. Although empirical validation and 49 intuitions were provided to support the working of these LR schedulers, a theoretical convergence 50 guarantee has not been provided to the best of our knowledge. 51

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

There is extensive research on the convergence behavior of perturbed SGD methods, where noise is added to the gradient during updates. In Jin et al. [15], the vanilla gradient descent is perturbed by samples from a ball whose radius is fixed using the optimization function-specific constants. They show escape from a saddle point by characterizing the distribution around a perturbed iterate as uniformly distributed over a perturbation ball along which the region corresponding to being stuck at a saddle point is shown to be very small. In Ge at al. [9], the saddle point escape for a perturbed stochastic gradient descent is proved using the second-order Taylor approximation of the optimization function, where the perturbation is applied from a unit ball to the stochastic gradient descent update. Following Ge at al. [9], several works prove the convergence of noisy stochastic gradient descent in the additive noise setting [33, 16, 2, 28]. In contrast to the above works which operate in the additive noise setting, our proposed LR scheduler results in multiplicative noise. Analyzing the convergence behavior under the new multiplicative noise setting is fairly challenging and results in a non-trivial addition to the literature.

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

78

79

80

81

82

83

84

85

86

87

## 1.1 Motivation

Traditional convergence analysis of gradient descent algorithms and its variants requires the use of a constant or a decaying learning rate [22]. However, with the introduction of LR schedulers, the learning rates are no longer monotonically decreasing. Rather, their values heavily fluctuate, with the occasional use of very large learning rates. Although there are ample justifications provided for the success of such methods, there are no theoretical results which prove that stochastic gradient descent algorithms with fluctuating learning rates converge to a local minimum in a non-convex setting. With the increase of emphasis on trustworthy artificial intelligence, we believe that it is important to no longer treat optimization algorithms as black-box models, and instead provide provable convergence guarantees while deviating from the proven classical implementation of the descent algorithms. In this work, we aim to bridge the gap by providing rigorous mathematical proof for the convergence of our proposed probabilistic LR scheduler with SGD.

## 1.2 Our contributions

1. We propose a new Probabilistic Learning Rate Scheduler (PLRS) where we model the learning rate as an instance of a random noise distribution.
2. We provide convergence proofs to show that SGD with our proposed PLRS converges to a local minimum in Section 4. To the best of our knowledge, we are the first to theoretically prove convergence of SGD with a LR scheduler that does not conform to constant or monotonically decreasing rates. We show how our LR scheduler, in combination with inherent SGD noise, speeds up convergence by escaping saddle points.
3. Our proposed probabilistic LR scheduler, while provably convergent, can be seamlessly ported into practice without the knowledge of theoretical constants (like gradient and Hessian-Lipschitz constants). We illustrate the efficacy of the PLRS through extensive

1 We abbreviate learning rate only in the context of learning rate scheduler as LR scheduler.

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

126

127

128

129

130

131

experimental validation, where we compare the accuracies with state-of-the-art schedulers in Section 5. We show that the proposed method outperforms popular schedulers such as cosine annealing [21], one-cycle [25], knee [14] and the multi-step scheduler when used with ResNet-110 on CIFAR-100, VGG-16 on CIFAR-10 and ResNet-50 on Tiny ImageNet, while displaying comparable performances on other architectures like DenseNet-40-12 and WRN-28-10 when trained on CIFAR-10 and CIFAR-100 datasets respectively. We also observe lesser spikes in the training loss across epochs which leads to a faster and more stable convergence. We provide our base code with all the hyperparameters for reproducibility in the supplemental material.

## 2 Probabilistic learning rate scheduler

Let f : R d → R be the function to be minimized. The unconstrained optimization, min x ∈ R d f ( x ) , can be solved iteratively using stochastic gradient descent whose update equation at time step t is given by

<!-- formula-not-decoded -->

Here, η t +1 ∈ R is the learning rate and g ( x t ) is the stochastic gradient of f ( x ) at time t . In this work, we propose a new LR scheduler, in which the learning rate η t +1 is sampled from a uniform random variable,

<!-- formula-not-decoded -->

Note that contrary to existing LR schedulers, which are deterministic functions, we propose that the learning rate at each time instant be a realization of a uniformly distributed random variable. Although the learning rate in our method is not scheduled, but is rather chosen as a random sample at every time step, we call our proposed method Probabilistic LR scheduler to keep in tune with the body of literature on LR schedulers. In order to represent our method in the conventional form of the stochastic gradient descent update, we split the learning rate η t +1 into a constant learning rate η c and a random component, as η t +1 = η c + u t +1 , where u t +1 ∼ U [ L min -η c , L max -η c ] . The stochastic gradient descent update using the proposed PLRS (referred to as SGD-PLRS) takes the form

<!-- formula-not-decoded -->

where we define w t as

<!-- formula-not-decoded -->

Here, ∇ f ( x t ) refers to the true gradient, i.e., ∇ f ( x t ) = E [ g ( x t )] . Note that in (3), the term x t -η c ∇ f ( x t ) resembles the vanilla gradient descent update and w t encompasses the noise in the update; the noise is inclusive of both the randomness due to the stochastic gradient as well as the randomness from the proposed LR scheduler. We set η c = L min + L max 2 so that the noise w t is zero mean, which we prove later in Lemma 1.

Remark 1. Note that a periodic LR scheduler such as triangular, or cosine annealing based scheduler can be considered as a single instance of our proposed PLRS. The range of values assigned to the learning rate η t +1 is pre-determined in both cases. In fact, for any LR scheduler, the basic mechanism is to vary the learning rate between a low and a high value - the high learning rates help escape the saddle point by perturbing the iterate, whereas the low values help in convergence. This pattern of switching between high and low values can be achieved through both stochastic and deterministic mechanisms. While the current literature explores the deterministic route (without providing analysis), we propose and explore the stochastic variant here and also provide a detailed analysis.

## 3 Preliminaries and definitions

We denote the Hessian of a function f : R d → R at x ∈ R d as H ( x ) := ∇ 2 f ( x ) and the minimum eigenvalue of the Hessian as λ min ( H ( x )) := λ min ( ∇ 2 f ( x )) respectively.

Definition 1. A function f : R d → R is said to be β -smooth (also referred to as β -gradient Lipschitz) if, ∃ β ≥ 0 such that,

<!-- formula-not-decoded -->

Definition 2. A function f : R d → R is said to be ρ -Hessian Lipschitz if, ∃ ρ ≥ 0 such that,

<!-- formula-not-decoded -->

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

164

165

166

167

168

169

170

171

172

Informally, a function is said to be gradient/Hessian Lipschitz, if the rate of change of the gradient/Hessian with respect to its input is bounded by a constant, i.e., the gradient/Hessian will not change rapidly. We now proceed to define approximate first and second-order stationary points of a given function f .

Definition 3. For a function f : R d → R that is differentiable, we say x ∈ R d is a ν - first-order stationary point ( ν -FOSP), if for a small positive value of ν , ∥∇ f ( x ) ∥ ≤ ν.

Before we define an ϵ -second order stationary point, we define a saddle point.

Definition 4. For a ρ -Hessian Lipschitz function f : R d → R that is twice differentiable, we say x ∈ R d is a saddle point if,

<!-- formula-not-decoded -->

where ν, γ &gt; 0 are arbitrary constants.

For a convex function, it is sufficient if the algorithm is shown to converge to the ν -FOSP as it would be the global minimum. However, in the case of a non-convex function, a point satisfying the condition for a ν -FOSP may not necessarily be a local minimum, but could be a saddle point or a local maximum. Hence, the Hessian of the function is required to classify it as a second-order stationary point, as defined below. Note that, in our analysis, we prove convergence of SGD-PLRS to the approximate second-order stationary point.

Definition 5. For a ρ -Hessian Lipschitz function f : R d → R that is twice differentiable, we say x ∈ R d is a ν -second-order stationary point ( ν -SOSP) if,

<!-- formula-not-decoded -->

where ν, γ &gt; 0 are arbitrary constants.

Definition 6. A function f : R d → R is said to possess the strict saddle property at all x ∈ R d if x fulfills any one of the following conditions: (i) ∥∇ f ( x ) ∥ ≥ ν , (ii) λ min ( H ( x )) ≤ -γ, (iii) x is close to a local minimum.

The strict saddle property ensures that an iterate stuck at a saddle point has a direction of escape.

Definition 7. A function f : R d → R is α -strongly convex if λ min ( H ( x )) ≥ α ∀ x ∈ R d .

We now provide the formal definitions of two common terms in time complexity.

Definition 8. A function f ( s ) is said to be O ( g ( s )) if ∃ a constant c &gt; 0 such that | f ( s ) | ≤ c | g ( s ) | . Here s ∈ S which is the domain of the functions f and g .

Definition 9. A function f ( s ) is said to be Ω( g ( s )) if ∃ a constant c &gt; 0 such that | f ( s ) | ≥ c | g ( s ) | .

In our analysis, we introduce the notations ˜ O ( . ) and ˜ Ω( . ) which hide all factors (including β , ρ , d , and α ) except η c , L min and L max in O and Ω respectively.

## 4 Proof of convergence

We present our convergence proofs to theoretically show that the proposed PLRS method converges to a ν -SOSP in finite time. We first state the assumptions that are instrumental for our proofs.

Assumptions 1. We now state the assumptions regarding the function f : R d → R that we require for proving the theorems.

- A1 The function f is β -smooth.
- A2 The function f is ρ -Hessian Lipschitz.
- A3 The norm of the stochastic gradient noise is bounded i.e, ∥ g ( x t ) -∇ f ( x t ) ∥ ≤ Q ∀ t ≥ 0 . Further, E [ Q 2 ] ≤ σ 2 .
- A4 The function f has strict saddle property.
- A5 The function f is bounded i.e., | f ( x ) | ≤ B, ∀ x ∈ R d .

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

- A6 The function f is locally α -strongly convex i.e, in the δ -neighborhood of a locally optimal point x ∗ for some δ &gt; 0 .

Remark 2. If ∇ ˜ f (˜ x t ) and ˜ g (˜ x t ) are the gradient and stochastic gradient of the second order Taylor approximation of f about the iterate ˜ x t , from Assumption A3 , it is implied that ∥ ∥ ∥ ˜ g (˜ x t ) -∇ ˜ f (˜ x t ) ∥ ∥ ∥ ≤ ˜ Q . Further, E [ ˜ Q 2 ] ≤ ˜ σ 2 .

Note that these assumptions are similar to those in the perturbed gradient literature [9, 15, 16]. We call attention to two significant differences in our approach compared to other perturbed gradient methods such as [15, 9, 16]: (i) In contrast to the isotropic additive perturbation commonly added to the SGD update, we introduce randomness in our learning rate, manifested as multiplicative noise in the update. This makes the characterization of the total noise dependent on the gradient, making the analysis challenging. (ii) The magnitude of noise injected is computed through the smoothness constants in the work by Jin et al. [15, 16]; instead, we treat the parameters L min and L max as hyperparameters to be tuned. This enables our PLRS method to be easily applied to training deep neural networks where the computation of these smoothness constants could be infeasible due to sheer computational complexity.

We reiterate the update equations of the proposed SGD-PLRS.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that the term w t has zero mean and we state this formally in the lemma below.

Lemma 1 (Zero mean property) . The mean of w t -1 ∀ t ≥ 1 is 0 .

Proof.

<!-- formula-not-decoded -->

This follows as E [ u t ] = L min + L max -2 η c 2 = 0 and E [ g ( x t -1 )] = ∇ f ( x t -1 ) .

For a function satisfying the Assumptions A1 -A6 , there are three possibilities for the iterate x t with respect to the function's gradient and Hessian, namely, B1: Gradient is large; B2: Gradient is small and iterate is around a saddle point; B3: Gradient is small and iterate is around a ν -SOSP.

We now present three theorems corresponding to each of these cases. Our first result pertains to the case B1 where the gradient of the iterate is large.

Theorem 1. Under the assumptions A1 and A3 with L max &lt; 1 β , for any point x t with ∥∇ f ( x t ) ∥ ≥ √ 3 η c βσ 2 where √ 3 η c βσ 2 &lt; ϵ , after one iteration, we have

<!-- formula-not-decoded -->

This theorem suggests that, for any iterate x t for which the gradient is large, the expected functional value of the subsequent iterate f ( x t +1 ) decreases, and the corresponding decrease E [ f ( x t +1 )] -f ( x t ) is in the order of ˜ Ω( L 2 max ) . The formal proof for this theorem can be found in Appendix A.

The next theorem corresponds to the case B2 where the gradient is small and the Hessian is negative. Theorem 2. Consider f satisfying Assumptions A1 -A5 . Let { x t } be the SGD iterates of the function f using PLRS. Let ∥∇ f ( x 0 ) ∥ ≤ √ 3 η c βσ 2 &lt; ϵ and λ min ( H ( x 0 )) ≤ -γ where ϵ, γ &gt; 0 . Then, there exists a T = ˜ O ( L -1 / 4 max ) such that with probability at least 1 -˜ O ( L 7 / 2 max ) ,

<!-- formula-not-decoded -->

The formal proof of this theorem is provided in Appendix C. The sketch of the proof is given below.

Proof Sketch This theorem shows that the iterates obtained using PLRS escape from a saddle point x 0 (where the gradient is small, and the Hessian has atleast one negative eigenvalue), i.e, it shows

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

248

249

250

251

252

253

254

the decrease in the expected value of the function f after T = ˜ O ( L -1 / 4 max ) iterations. Note that for a ρ -Hessian smooth function,

<!-- formula-not-decoded -->

To evaluate E [ f ( x T ) -f ( x 0 )] from (9), we require an analytical expression for x T -x 0 , which is not tractable. Hence, we employ the second-order Taylor approximation of the function f , which we denote as ˜ f . We then apply SGD-PLRS on ˜ f to obtain ˜ x T . Following this, we write x T -x 0 = ( x T -˜ x T ) + (˜ x T -x 0 ) and derive expressions for upper bounds on ˜ x T -x 0 and x T -˜ x T which hold with high probability in Lemmas 2 and 3, respectively (given in Appendix B.1 and B.2).

We split the quadratic term in (9) into two parts corresponding to ˜ x T -x 0 and x T -˜ x T . We further decompose the term, say Y = (˜ x T -x 0 ) T H ( x 0 )(˜ x T -x 0 ) into its eigenvalue components along each dimension with corresponding eigenvalues λ 1 , . . . , λ d of H ( x 0 ) . Our main result in this theorem proves that the term Y dominates over all the other terms of (9), and that it is bounded by a negative value, thereby, proving E [ f ( x T )] ≤ f ( x 0 ) . This main result uses a two-pronged proof. Firstly, we use our assumption that the initial iterate x 0 is at a saddle point and hence at least one of λ i , 1 ≤ i ≤ d is negative. We formally show that the eigenvector corresponding to this eigenvalue points to the direction of escape. Secondly, we use the second order statistics of our noise, to show that the magnitude of Y is large enough to dominate over the other terms of (9). Note that our noise term involves the stochasticity in the gradient and the probabilistic learning rate. Hence, we have shown that the negative eigenvalue of the Hessian at a saddle point and the unique characterization of the noise is sufficient to force a descent along the negative curvature safely out of the region of the saddle point within T iterations.

As each SGD-PLRS update is noisy, we need to ensure that once we escape a saddle point and move towards a local minimum (case B3 ), we do not overshoot the minimum but rather, stay in the δ -neighborhood of an SOSP, with high probability. We formalize this in Theorem 3.

Theorem 3. Consider f satisfying the assumptions A1 -A6 . Let the initial iterate x 0 be δ close to a local minimum x ∗ such that ∥ x 0 -x ∗ ∥ ≤ ˜ O ( √ L max ) &lt; δ . With probability at least 1 -ξ , ∀ t ≤ T where T = ˜ O ( 1 L 2 max log 1 ξ ) ,

<!-- formula-not-decoded -->

This theorem deals with the case that the initial iterate x 0 is δ -close to a local minimum x ∗ (case B3 ). We prove that the subsequent iterates are also in the same neighbourhood, i.e., δ close to the local minimum, with high probability. In other words, we prove that the sequence {∥ x t -x ∗ ∥} is bounded by δ for t ≤ T . In the neighbourhood of the local minimum, gradients are small and subsequently, the change in iterates, x t -x t -1 are minute. Therefore, the iterates stay near the local minimum with high probability. It is worth noting that the nature of the noise, which is comprised of stochastic gradients (whose stochasticity is bounded by Q ) multiplied with a bounded uniform random variable (owing to PLRS), aids in proving our result. We provide the formal proof in Appendix D.

## 5 Empirical evaluation

We provide results on CIFAR-10, CIFAR-100 [18] and Tiny ImageNet [20] and compare with: (i) cosine annealing with warm restarts [21], (ii) one-cycle scheduler [25], (iii) knee scheduler [14], (iv) constant learning rate and (v) multi-step decay scheduler. We run experiments for 500 epochs for the CIFAR datasets and for 100 epochs for the Tiny ImageNet dataset using the SGD optimizer for all schedulers 2 . We also set all other regularization parameters, such as weight decay and dampening, to zero. We use a batch size of 64 for DenseNet-40-12, 50 for ResNet-50, and 128 for the others. We conduct all our experiments in a single NVIDIA GeForce RTX 2080 GPU card.

To determine the parameters L min and L max for PLRS, we perform a range test, where we observe the training loss for a range of learning rates as is done in state-of-the-art LR schedulers such as

2 We provide results without momentum to be consistent with our theoretical framework. When we used the SGD optimizer with momentum for PLRS, we obtain results better than those reported without momentum.

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

283

284

285

286

287

288

289

290

291

Table 1: Maximum and mean (with standard deviation) test accuracies over 3 runs for CIFAR-10.

| Architecture   | Scheduler   |   Max acc. | Mean acc. (S.D)   |
|----------------|-------------|------------|-------------------|
| VGG-16         | Cosine      |      96.87 | 96.09 (0.78)      |
| VGG-16         | Knee        |      96.87 | 96.35 (0.45)      |
| VGG-16         | One-cycle   |      90.62 | 89.06 (1.56)      |
| VGG-16         | Constant    |      96.09 | 96.06 (0.05)      |
| VGG-16         | Multi-step  |      92.97 | 92.45 (0.90)      |
| VGG-16         | PLRS (ours) |      97.66 | 96.09 (1.56)      |
| WRN-28-10      | Cosine      |      92.03 | 91.90 (0.13)      |
| WRN-28-10      | Knee        |      92.04 | 91.64 (0.63)      |
| WRN-28-10      | One-cycle   |      87.76 | 87.37 (0.35)      |
| WRN-28-10      | Constant    |      92.04 | 92.00 (0.08)      |
| WRN-28-10      | Multi-step  |      88.94 | 88.80 (0.21)      |
| WRN-28-10      | PLRS (ours) |      92.02 | 91.43 (0.54)      |

one-cycle [25] and knee schedulers [14]. As the learning rate is gradually increased, we first observe a steady decrease in the training loss, then followed by a drastic increase. We note the learning rate at which there is an increase of training loss, say ¯ L and choose the maximum learning rate L max to be just below ¯ L , where the loss is still decreasing. We then tune L min such that 0 &lt; L min &lt; L max . We choose the parameters for the baseline schedulers as suggested in the original papers (further details of parameters are provided in Appendix F).

While there are ample works which prove the convergence of SGD with additive noise as in [9], they cannot be ported into practice for deep neural networks. They require smoothness constants [33, 16, 2] or functional bounds on the norms of the function derivatives [28] to be computed for the additive noise injection, which can not be obtained for the loss functions of neural networks or can only be approximated locally [19]. Further, the empirical convergence properties of noisy SGD are not demonstrated through examples in the majority of these analytical works which makes it hard to compare their convergence with PLRS. However, we compare our proposed PLRS against the noisy SGD mechanism proposed by Ge et al. [9] providing convergence results on the online tensor decomposition problem using the code provided by the authors in Appendix G.

## 5.1 Results for CIFAR-10

We consider VGG-16 [23] and WRN-28-10 [29] and use L min = 0 . 07 and L max = 0 . 1 for both the networks. We record the maximum and mean test accuracies across different LR schedulers in Table 1. The highest accuracy across schedulers is recorded in bold. For the VGG-16 network, we rank the highest in terms of maximum test accuracy. In terms of the mean test accuracy over 3 runs, the knee scheduler outperforms the rest. Note that the second highest mean test accuracy is achieved by both PLRS and the cosine annealing schedulers. Unsurprisingly, the constant scheduler has the lowest standard deviation. In the WRN-28-10 network, note that the maximum test accuracies for the cosine, knee, constant and the PLRS schedulers are very similar (difference in the order of 10 -2 ). Their similar performance is also reflected in the mean test accuracies although the constant learning rate edges out the other schedulers marginally. To study the convergence of the schedulers we also plot the training loss across epochs in Figure 1. We observe that our proposed PLRS achieves one of the fastest rates of convergence in terms of the training loss compared across all the schedulers for both networks. Note that the cosine annealing scheduler records several spikes across the training.

## 5.2 Results for CIFAR-100

We consider networks ResNet-110 [10] and DenseNet-40-12 [13], and use L min = 0 . 07 and L max = 0 . 1 for the former, and L min = 0 . 1 and L max = 0 . 2 for the latter. The maximum and the mean test accuracies (with standard deviation) across 3 runs are provided in Table 2. For ResNet-110, PLRS performs best in terms of the maximum and the mean test accuracies. This is closely followed by the other state-of-the-art LR schedulers such as knee and cosine schedulers. For the DenseNet-4012 network, PLRS comes to a close second to the multi-step LR scheduler in terms of the maximum and mean test accuracies. However, it is important to note that the multi-step scheduler records the

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

3.0

Figure 1: Training loss vs epochs for VGG-16 and WRN-28-10 for CIFAR-10.

<!-- image -->

Table 2: Maximum and mean (with standard deviation) test accuracies over 3 runs for CIFAR-100.

| Architecture   | Scheduler   |   Max acc. | Mean acc.(S.D)   |
|----------------|-------------|------------|------------------|
| ResNet-110     | Cosine      |      74.22 | 72.66 (1.56)     |
| ResNet-110     | Knee        |      75.78 | 72.39 (2.96)     |
| ResNet-110     | One-cycle   |      71.09 | 70.05 (1.19)     |
| ResNet-110     | Constant    |      69.53 | 66.67 (2.51)     |
| ResNet-110     | Multi-step  |      63.28 | 61.20 (2.39)     |
| ResNet-110     | PLRS (ours) |      77.34 | 74.61 (2.95)     |
| DenseNet-40-12 | Cosine      |      82.81 | 80.47 (2.07)     |
| DenseNet-40-12 | Knee        |      82.81 | 80.73 (2.39)     |
| DenseNet-40-12 | One-cycle   |      73.44 | 72.39 (0.90)     |
| DenseNet-40-12 | Constant    |      82.81 | 80.73 (2.39)     |
| DenseNet-40-12 | Multi-step  |      87.5  | 84.89 (2.39)     |
| DenseNet-40-12 | PLRS (ours) |      84.37 | 83.33 (0.90)     |

least test accuracy with the ResNet-110 network. Hence, its performance is not consistent across the networks, while PLRS is consistently one of the best performing schedulers.

We plot the training loss in Figure 2. For ResNet-110, both PLRS and knee LR scheduler converge to a low training loss around 150 epochs. While cosine annealing LR scheduler also seems to converge fast, it experiences sharp spikes along the curve during the restarts. For DenseNet-40-12, PLRS converges faster to a lower training loss compared to the other schedulers.

## 5.3 Results for Tiny ImageNet

We consider the Resnet-50 [10] architecture and use L min = 0 . 35 and L max = 0 . 4 . We present the maximum and mean test accuracies in Table 3. We provide the plot of training loss in Figure 3. PLRS performs the best in terms of maximum test accuracy. In terms of mean test accuracy, it ranks second next to cosine annealing by a close margin. It can be observed that PLRS achieves the fastest convergence to the lowest training loss compared to others. Moreover, it exhibits stable convergence, especially when compared cosine annealing, which experiences multiple spikes due to warm restarts.

## 5.4 Limitations and broader impact

In line with all other works which focus on convergence proofs, our work too applies only to a restricted class of functions that meet the assumptions in Section 4. In contrast, our experiments are conducted on deep neural networks, which may not strictly satisfy these assumptions. While this is a limitation of our work, we note that many papers focused on theoretical convergence of SGD do not include empirical results, and many practice-oriented papers proposing new LR schedulers do

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

<!-- image -->

Figure 2: Training loss vs epochs for ResNet-110 and DenseNet-40-12 on CIFAR-100.

Table 3: Maximum and mean (with standard deviation) test accuracies over 3 runs for Tiny ImageNet.

| Scheduler   |   Max acc. | Mean acc. (S.D)   |
|-------------|------------|-------------------|
| Cosine      |      62.13 | 62.03 (0.15)      |
| Knee        |      61.93 | 61.50 (0.42)      |
| One-cycle   |      52.24 | 51.99 (0.22)      |
| Constant    |      61.59 | 61.11 (0.42)      |
| Multi-step  |      61.28 | 61.20 (0.08)      |
| PLRS (ours) |      62.34 | 61.90 (0.73)      |

Figure 3: Training loss vs epochs for ResNet-50 with Tiny ImageNet.

<!-- image -->

not include convergence proofs. Another limitation is that our experiments are limited to benchmark image datasets, even though our proposed scheduler is general and can be applied to other domains.

Our work contributes to the relatively underexplored theoretical understanding of LR schedulers, an area where most prior research has focused on empirical or application-driven results. As discussed earlier, commonly used periodic schedulers, such as triangular or cosine annealing, can be viewed as special cases of our proposed PLRS. This generalization opens new avenues for theoretical investigation, including the analysis of convergence properties across a broader class of schedulers. In practice, PLRS demonstrates improved stability and enables faster convergence, reducing the number of training epochs required. This efficiency translates to lower GPU usage and energy consumption, supporting more sustainable and resource-conscious AI development.

## 6 Concluding remarks

We have proposed the novel idea of a probabilistic LR scheduler. The probabilistic nature of the scheduler helped us provide the first theoretical convergence proofs for SGD using LR schedulers. In our opinion, this is a significant step in the right direction to bridge the gap between theory and practice in the LR scheduler domain. Our empirical results show that our proposed LR scheduler performs competitively with the state-of-the-art cyclic schedulers, if not better, on CIFAR-10, CIFAR100, and Tiny ImageNet datasets for a wide variety of popular deep architectures. This leads us to hypothesize that the proposed probabilistic LR scheduler acts as a super-class of LR schedulers encompassing both probabilistic and deterministic schedulers. Future research directions include further exploration of this hypothesis.

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

369

370

371

372

373

374

375

376

377

## References

- [1] Maksym Andriushchenko and Nicolas Flammarion. Understanding and improving fast adversarial training. In Advances in Neural Information Processing Systems , volume 33, pages 16048-16059, 2020.
- [2] Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Nathan Srebro, and Blake Woodworth. Lower bounds for non-convex stochastic optimization. Mathematical Programming , 199(1):165-214, 2023.
- [3] James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of machine learning research , 13(2):281-305, 2012.
- [4] James Bergstra, Daniel Yamins, and David Cox. Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures. In International conference on machine learning , pages 115-123, 2013.
- [5] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision , pages 9650-9660, 2021.
- [6] Yann N Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio. Identifying and attacking the saddle point problem in high-dimensional nonconvex optimization. In Advances in neural information processing systems , volume 27, page 2933-2941, 2014.
- [7] Guneet Singh Dhillon, Pratik Chaudhari, Avinash Ravichandran, and Stefano Soatto. A baseline for few-shot image classification. In International Conference on Learning Representations , 2020.
- [8] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research , 12(7):2121-2159, 2011.
- [9] Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points: Online stochastic gradient for tensor decomposition. Journal of Machine Learning Research , 40:1-46, 2015.
- [10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016.
- [11] Wassily Hoeffding. Probability inequalities for sums of bounded random variables. The collected works of Wassily Hoeffding , pages 409-426, 1994.
- [12] Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics , pages 328-339, Melbourne, Australia, 2018.
- [13] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 4700-4708, 2017.
- [14] Nikhil Iyer, V Thejas, Nipun Kwatra, Ramachandran Ramjee, and Muthian Sivathanu. Wideminima density hypothesis and the explore-exploit learning rate schedule. Journal of Machine Learning Research , 24(65):1-37, 2023.
- [15] Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M Kakade, and Michael I Jordan. How to escape saddle points efficiently. In International conference on machine learning , pages 1724-1732. PMLR, 2017.
- [16] Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M. Kakade, and Michael I. Jordan. On nonconvex optimization for machine learning: Gradients, stochasticity, and saddle points. Journal of the Association for Computing Machinery , 68(2):1-29, 2021.

- [17] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint 378 arXiv:1412.6980 , 2014. 379
- [18] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 380 Technical Report TR-2009, University of Toronto, Toronto, ON, Canada, 2009. 381
- [19] Fabian Latorre, Paul Thierry Yves Rolland, and Volkan Cevher. Lipschitz constant estimation 382 for neural networks via sparse polynomial optimization. In 8th International Conference on 383 Learning Representations , 2020. 384

385

386

- [20] Ya Le and Xuan Yang. Tiny imagenet visual recognition challenge. [Online]. Available: https://tinyimagenet.herokuapp.com, 2015.

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

417

418

419

420

421

422

- [21] Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts. In 5th International Conference on Learning Representations , 2017.
- [22] Yurii Nesterov. Introductory Lectures on Convex Optimization: A Basic Course . Springer Publishing Company, Incorporated, 2014.
- [23] K Simonyan and A Zisserman. Very deep convolutional networks for large-scale image recognition. In 3rd International Conference on Learning Representations , 2015.
- [24] Leslie N Smith. Cyclical learning rates for training neural networks. In 2017 IEEE winter conference on applications of computer vision , pages 464-472, 2017.
- [25] Leslie N Smith and Nicholay Topin. Super-convergence: Very fast training of neural networks using large learning rates. In Artificial intelligence and machine learning for multi-domain operations applications , volume 11006, pages 369-386, 2019.
- [26] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 1-9, 2015.
- [27] Thulasi Tholeti and Sheetal Kalyani. Tune smarter not harder: A principled approach to tuning learning rates for shallow nets. IEEE Transactions on Signal Processing , 68:5063-5078, 2020.
- [28] Daniel Yiming Cao, August Y Chen, Karthik Sridharan, and Benjamin Tang. Efficiently escaping saddle points under generalized smoothness via self-bounding regularity. arXiv e-prints , pages arXiv-2503, 2025.
- [29] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In British Machine Vision Conference 2016 . British Machine Vision Association, 2016.
- [30] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 5728-5739, 2022.
- [31] Matthew D Zeiler. Adadelta: an adaptive learning rate method. arXiv preprint arXiv:1212.5701 , 2012.
- [32] Jingfeng Zhang, Bo Han, Laura Wynter, Bryan Kian Hsiang Low, and Mohan Kankanhalli. Towards robust resnet: a small step but a giant leap. In Proceedings of the 28th International Joint Conference on Artificial Intelligence , pages 4285-4291, 2019.
- [33] Yuchen Zhang, Percy Liang, and Moses Charikar. A hitting time analysis of stochastic gradient langevin dynamics. In Conference on Learning Theory , pages 1980-2022. PMLR, 2017.
- [34] Pan Zhou, Jiashi Feng, Chao Ma, Caiming Xiong, Steven Hoi, and E. Weinan. Towards theoretically understanding why sgd generalizes better than adam in deep learning. In Advances in Neural Information Processing Systems , pages 16048-16059, 2020.

425

426

427

428

429

430

431

## A Proof of Theorem 1

Theorem 4 (Theorem 1 restated) . Under the assumptions A1 and A3 with L max &lt; 1 β , for any point x t with ∥∇ f ( x t ) ∥ ≥ √ 3 η c βσ 2 where √ 3 η c βσ 2 &lt; ϵ (satisfying B1 ), after one iteration we have,

<!-- formula-not-decoded -->

Proof. Using the second order Taylor series approximation for f ( x t +1 ) around x t , where x t +1 = x t -η c ∇ f ( x t ) -w t , we have

<!-- formula-not-decoded -->

following the result from [22, Lemma 1.2.3]. Taking expectation w.r.t. w t , 432

<!-- formula-not-decoded -->

since E [ w t ] = 0 due to the zero mean property in Lemma 1. We focus on the last term in the next 433 steps. Expanding ∥ w t ∥ 2 , 434

<!-- formula-not-decoded -->

Taking expectation with respect to x t and noting that E [ u t +1 ] = 0 and E [ g ( x t )] = ∇ f ( x t ) , 3 435

<!-- formula-not-decoded -->

Now, as per assumption A3 , 436

<!-- formula-not-decoded -->

as E [ Q 2 ] ≤ σ 2 . Applying (12) to (11), 437

<!-- formula-not-decoded -->

3 Note that there are two random variables in w t which are the stochastic gradient g ( x t ) and the uniformly distributed LR u t +1 due to our proposed LR scheduler. Hence, the expectation is with respect to both these variables. Also note that u t +1 and g ( x t ) are independent of each other.

## Appendix

since the second moment of a uniformly distributed random variable in the interval [ L min -η c , L max -438 η c ] is given by ( L max -L min ) 2 12 . Using (13) in (10) and η c = L min + L max 2 , 439

<!-- formula-not-decoded -->

Now, applying our initial assumption that ∥∇ f ( x t ) ∥ ≥ √ 3 η c βσ 2 , we have, 440

<!-- formula-not-decoded -->

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

Since L max &lt; 1 β and η c = L min + L max 2 , we have η c β &lt; L max β &lt; 1 . Finally,

<!-- formula-not-decoded -->

which proves the theorem.

## B Additional results needed to prove Theorem 2

Here, we state and prove two lemmas that are instrumental in the proof of Theorem 2.

## B.1 Proof of Lemma 2

In the following Lemma, we prove that the gradients of a second order approximation of f are probabilistically bounded for all t ≤ T and its iterates as we apply SGD-PLRS are also bounded when the initial iterate x 0 is a saddle point.

Lemma 2. Let f satisfy Assumptions A1 -A4 . Let ˜ f be the second order Taylor approximation of f and let ˜ x t be the iterate at time step t obtained using the SGD update equation as in (3) on ˜ f ; let ˜ x 0 = x 0 , ∥∇ f ( x 0 ) ∥ ≤ ϵ and the minimum eigenvalue of the Hessian of f at x 0 be λ min ( H ( x 0 )) = -γ o where γ o &gt; 0 . With probability at least 1 -˜ O ( L 15 / 4 max ) , we have

<!-- formula-not-decoded -->

Proof. As ˜ f is the second order Taylor series approximation of f , we have 453

<!-- formula-not-decoded -->

Taking derivative w.r.t. ˜ x , we have ∇ ˜ f (˜ x ) = ∇ f ( x 0 ) + H ( x 0 )(˜ x -x 0 ) . Now, note that ∇ ˜ f (˜ x t -1 ) = 454 ∇ f ( x 0 ) + H ( x 0 )(˜ x t -1 -x 0 ) = K ( x 0 ) + H ( x 0 )˜ x t -1 , where K ( x 0 ) = ∇ f ( x 0 ) -H ( x 0 ) x 0 = 455 ∇ ˜ f (˜ x t -1 ) -H ( x 0 )˜ x t -1 . Therefore, 456

<!-- formula-not-decoded -->

Next, using the SGD-PLRS update and rearranging, 457

<!-- formula-not-decoded -->

where I denotes the d × d identity matrix. Next, unrolling the term ∇ ˜ f (˜ x t -1 ) recursively, 458

<!-- formula-not-decoded -->

Using the triangle and Cauchy-Schwartz inequalities, 459

<!-- formula-not-decoded -->

Note that the norm over the matrices refers to the matrix-induced norm. Since H ( x 0 ) is a real 460 symmetric matrix, the induced norm gives the maximum eigenvalue of H ( x 0 ) i.e, λ max ( H ( x 0 )) ≤ 461 β by our β -smoothness assumption A1 . In the case of ( I -η c H ( x 0 )) the induced norm gives 462 (1 -η c λ min ( H ( x 0 )) which is (1 + η c γ o ) as per our assumption that λ min ( H ( x 0 )) = -γ o . Also 463 recall that ∥ ∥ ∥ ∇ ˜ f (˜ x 0 ) ∥ ∥ ∥ ≤ ϵ . Now (17) becomes, 464

<!-- formula-not-decoded -->

Now, expanding the noise term ˜ w τ , 465

<!-- formula-not-decoded -->

Now recall from our assumption A3 that ∥ ∥ ∥ ˜ g (˜ x τ ) -∇ ˜ f (˜ x τ ) ∥ ∥ ∥ ≤ ˜ Q . Hence, 466

<!-- formula-not-decoded -->

Using ∥ ∥ ∥ ∇ ˜ f (˜ x 0 ) ∥ ∥ ∥ ≤ ϵ and ∥ ∥ ∥ ∇ ˜ f (˜ x 1 ) ∥ ∥ ∥ ≤ (1 + η c γ o ) ϵ + ϵ +2 ˜ Q , it can be proved by induction that 467 the general expression for t ≥ 2 is given by, 468

<!-- formula-not-decoded -->

We give the proof of (19) by induction in Appendix E. Next, we prove the bound on ˜ x t -˜ x 0 . Using 469 the SGD-PLRS update, 470

<!-- formula-not-decoded -->

where the equation (20a) is obtained by using (16). We obtain (20b) by using the summation of 471 geometric series as H ( x 0 ) is invertible by the strict saddle property. As ˜ x 0 = x 0 , we can write 472 ∇ ˜ f (˜ x 0 ) = ∇ f ( x 0 ) . Taking norm, 473

<!-- formula-not-decoded -->

In (21), it can be seen that the first term is arbitrarily small by the initial assumption and that the 474 second term decides the order of ∥ ˜ x t -˜ x 0 ∥ . Hence, in order to bound ∥ ˜ x t -˜ x 0 ∥ probabilistically, it 475 is sufficient to bound the second term, ∑ t -1 τ =0 (1 + η c γ o ) t -τ -1 ∥ ˜ w τ ∥ . Now, 476

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

478

479

480

481

482

483

484

It can be observed from (22) that the last term dominates the expression of and hence, it determines the order of ∥ ˜ x t -˜ x 0 ∥ . We now apply Hoeffding's inequality to derive a probabilistic bound on ∥ ˜ x t -˜ x 0 ∥ . According to Hoeffding's inequality for any summation S n = X 1 + · · · + X n such that a i ≤ X i ≤ b i , P ( S n -E [ S n ] ≥ δ ) ≤ exp ( -2 δ 2 ∑ n i =1 ( b i -a i ) 2 ) . Now, setting T = ˜ O ( L -1 / 4 max ) from (41) and assuming η c ≤ η max ≤ √ 2 -1 γ ′ , γ o ≤ γ ′ , the squared bound of the summation ∑ t -1 τ =2 (1 + η c γ o ) t -τ -1 | u τ +1 | 10 ˜ Q ∑ τ ( τ -1) 2 τ ′ =0 (1 + η c γ o ) τ ′ ≤ ˜ O ( L 3 / 4 max ) , Setting δ = ˜ O (√ L 3 / 4 max log ( 1 L max ) ) , for some t ≤ T ,

<!-- formula-not-decoded -->

Taking the union bound over all t ≤ T , 485

<!-- formula-not-decoded -->

which completes our proof. 486

## B.2 Proof of Lemma 3 487

488

489

This lemma is used to derive an expression for a high probability upper bound of ∥ x t -˜ x t ∥ and ∥ ∥ ∥ ∇ f ( x t ) -∇ ˜ f (˜ x t ) ∥ ∥ ∥ .

490

491

492

493

494

Lemma 3. Let f : R d → R satisfy Assumptions A1 -A4 . Let ˜ f be the second order Taylor's approximation of f and let x t , ˜ x t be the iterates at time step t obtained using the SGD-PLRS update on f , ˜ f respectively; let ˜ x 0 = x 0 and ∥∇ f ( x 0 ) ∥ ≤ ϵ . Let the minimum eigenvalue of the Hessian at x 0 be λ min ( ∇ 2 ( f ( x 0 ))) = -γ o , where γ o &gt; 0 . Then ∀ t ≤ T = O ( L -1 / 4 max ) , with a probability of at least 1 -˜ O ( L 7 / 2 max ) ,

<!-- formula-not-decoded -->

Proof. The expression for x t -˜ x t can be written as, 495

<!-- formula-not-decoded -->

where we define ∆ t = ∇ f ( x t ) -∇ ˜ f (˜ x t ) . Now in order to bound ∥ x t -˜ x t ∥ , we derive expressions 496 for both w τ -˜ w τ and ∆ τ . We initially focus on the term w τ -˜ w τ . 497

<!-- formula-not-decoded -->

Taking norm on both sides, 498

<!-- formula-not-decoded -->

Using (24) and (25) in (23), and assumption A3 that stochastic noise is bounded, and applying norm, 499

<!-- formula-not-decoded -->

Next, we focus on providing a bound for ∥ ∆ t ∥ . Recall that ∆ t = ∇ f ( x t ) -∇ ˜ f (˜ x t ) . The gradient 500 can be written as [22], 501

<!-- formula-not-decoded -->

where θ t -1 = ( ∫ 1 0 ( H ( x t -1 + v ( x t -x t -1 )) -H ( x t -1 ) ) dv ) ( x t -x t -1 ) . Let H ′ t -1 = H ( x t -1 ) -502 H ( x 0 ) . Using the SGD-PLRS update, 503

<!-- formula-not-decoded -->

From (14) in the proof of Lemma 2, 504

<!-- formula-not-decoded -->

Subtracting (28) from (27), we obtain ∆ t as, 505

<!-- formula-not-decoded -->

We now have an expression for ∆ t . However, the derived expression is recursive and contains ∆ t -1 . 506 We focus on eliminating the recursive dependence and obtain a stand-alone bound for ∥ ∆ t ∥ ∀ t ≤ T . 507 Now, we bound each of the five terms (we term them T 1 , · · · , T 5 ) of (29). First, let us define the 508 events, 509

<!-- formula-not-decoded -->

It can be seen that R t ⊂ R t -1 and C t ⊂ C t -1 . Note that, from Lemma 2, we know the probabilistic 510 characterization of R t . We comment on the parameter µ later in the proof. Now, we derive bounds 511 for each term of ∆ t conditioned on the event R t -1 ∩ C t -1 for time t ≤ T = O ( L -1 / 4 max ) . 512

<!-- formula-not-decoded -->

where (30) follows from the definition of event C t -1 . Note that the first term in (30) governs the 513 order of the expression (as 0 ≤ L max ≤ 1 ). 514

<!-- formula-not-decoded -->

where the substitution follows from (25). To bound T 3 and T 4 , we first bound H ′ t -1 , 515

<!-- formula-not-decoded -->

where (31a) follows from the assumption A2 while (31b) follows from (26). We use the bounds 516 defined for events R t -1 ∩ C t -1 in (31b) and (31c). Now, using the bound for ∥ ∥ ∥ H ′ t -1 ∥ ∥ ∥ , T 3 can be 517

bounded as follows. 518

<!-- formula-not-decoded -->

where we use the bounds in the event R t -1 ∩ C t -1 and (31d). 519

<!-- formula-not-decoded -->

where we use assumption A3 in (32a) and the bounds of R t -1 ∩ C t -1 and (31d) in (32b). 520

<!-- formula-not-decoded -->

Here, we use assumption A3 and the bounds of the event R t -1 ∩ C t -1 in (33b). Note that we have 521 derived bounds so far conditioned on the event R t -1 ∩ C t -1 . We now include this conditioning 522 explicitly in our notations going forward. 523

To characterize ∥ ∆ t ∥ 2 , we construct a supermartingale process; and to do so, we focus on finding 524 E [ ∥ ∆ t ∥ 2 1 R t -1 ∩ C t -1 ] using the bounds derived for the terms T 1 , · · · , T 5 . Later, we use the Azuma525

Hoeffding inequality to obtain a probabilistic bound of ∥ ∆ t ∥ . 526

<!-- formula-not-decoded -->

Now, let 527

<!-- formula-not-decoded -->

Now, in order to prove the process G t 1 R t -1 ∩ C t -1 is a supermartingale, we prove that 528 E [ G t 1 R t -1 ∩ C t -1 | S t -1 ] ≤ G t -1 1 R t -2 ∩ C t -2 . We define a filtration S t = s { w 0 , . . . , w t -1 } where 529 s { . } denotes a sigma-algebra field. 530

<!-- formula-not-decoded -->

To obtain (36a), we use (34) to find E [ G t 1 R t -1 ∩ C t -1 | S t -1 ] . In (36b), we upper bound by the 531 multiplication of a positive term (1 + η c γ o ) 2 . Therefore, G t 1 R t -1 ∩ C t -1 is a supermartingale. 532

<!-- formula-not-decoded -->

533

534

535

536

537

538

Note that the above expression is obtained by the observation that the only random terms of ∆ t conditioned on the filtration S t -1 = s { w 0 , w 1 , . . . , w t -2 } are H ( x 0 ) ( w t -1 -˜ w t -1 ) , H ′ t -1 w t -1 and θ t -1 (see (33a)). Hence, we cancel out the deterministic terms in ∥ ∆ t ∥ 2 and E ∥ ∆ t ∥ 2 and neglect the negative terms while upper bounding.

The Azuma-Hoeffding inequality for martingales and supermartingales [11] states that if { G t 1 R t -1 ∩ C t -1 } is a supermartingale and | G t 1 R t -1 ∩ C t -1 -G t -1 1 R t -2 ∩ C t -2 | ≤ c t almost surely,

then for all positive integers t and positive reals δ , 539

<!-- formula-not-decoded -->

The bound of | G t 1 R t -1 ∩ C t -1 -G t -1 1 R t -2 ∩ C t -2 | can be obtained using the definition of the process 540 G t in (35). Recollecting our assumption that η c ≤ η max ≤ √ 2 -1 γ ′ , γ o ≤ γ ′ , we see that (1 + 541 η c γ o ) -2 t ≤ ˜ O (1) . Therefore, 542

<!-- formula-not-decoded -->

We denote the bound obtained for | G t 1 R t -1 ∩ C t -1 -E [ G t 1 R t -1 ∩ C t -1 | S t -1 ] | as c t -1 . Now, let 543 δ = √ ∑ t -1 τ =0 c 2 τ log 1 L max in the Azuma-Hoeffding inequality. Now, for any t ≤ T = O ( L -1 / 4 max ) , 544

<!-- formula-not-decoded -->

After taking union bound ∀ t ≤ T , 546

<!-- formula-not-decoded -->

We represent the hidden constants in ˜ O ( µL 3 / 4 max log 2 1 L max ) by ˜ c and choose µ such that µ &lt; ˜ c . 547 Then, the following equation holds true. 548

<!-- formula-not-decoded -->

549

Hence we can write,

<!-- formula-not-decoded -->

We need the probability of the event C t , ∀ t ≤ T in order to prove the lemma. From Lemma 2, we get 550 the probability of the event ¯ R t as ˜ O ( L 15 / 4 max ) . Then, 551

<!-- formula-not-decoded -->

where the first term of (38) follows from (37). The second term of (38) can be bounded by P ( ¯ R t -1 ) 552 which is known by Lemma 2. Finally, 553

<!-- formula-not-decoded -->

The probability P C t -1 ) can be found as, 554

<!-- formula-not-decoded -->

As T = O ( L -1 / 4 max ) , P ( ¯ C T ) ≤ ˜ O ( L 7 / 2 max ) . From (26), 555

∥ x t -˜ x t ∥ ≤ t -1 ∑ τ =0 ( η c + | u τ +1 | ) ( ∥ ∆ τ ∥ + Q + ˜ Q ) ≤ O ( 1 L 1 / 4 max )( ˜ O ( L max ) µL 3 / 8 max log 1 L max + ˜ O ( L max ) ) = O ( µL 9 / 8 max log 1 L max ) + ˜ O ( L 3 / 4 max ) ≤ ˜ O ( L 3 / 4 max ) This completes our proof. 556

557

558

559

560

561

562

563

564

565

566

567

568

569

570

571

572

573

574

## C Proof of Theorem 2

Theorem 5. (Theorem 2 restated) Consider f satisfying Assumptions A1 -A5 . Let f be the second order Taylor approximation of f ; let { x t } and { ˜ x t } be the corresponding SGD iterates using PLRS, with ˜ x 0 = x 0 . Let x 0 correspond to B2 , i.e., ∥∇ f ( x 0 ) ∥ ≤ ϵ and λ min ( H ( x 0 )) ≤ -γ where ϵ, γ &gt; 0 . Then, there exists a T = ˜ O ( L -1 / 4 max ) such that with probability at least 1 -˜ O ( L 7 / 2 max ) ,

<!-- formula-not-decoded -->

Proof. In this proof, we consider the case when the initial iterate x 0 is at a saddle point (corresponding to B2 ). This theorem shows that the SGD-PLRS algorithm escapes the saddle point in T steps where T = ˜ O ( L -1 / 4 max ) .

We use the Taylor series approximation in order to make the problem tractable. Similar to the SGD-PLRS updates for the function f , the SGD update on the function ˜ f can be given as,

<!-- formula-not-decoded -->

As the function f is ρ -Hessian, using [22, Lemma 1.2.4] and the Taylor series expansion one obtains, f ( x ) ≤ f ( x 0 ) + ∇ f ( x 0 ) T ( x -x 0 ) + 1 2 ( x -x 0 ) T H ( x 0 )( x -x 0 ) + ρ 6 ∥ x -x 0 ∥ 3 . Let ˜ κ = ˜ x T -x 0 , κ = x T -˜ x T . Note that ˜ κ + κ = x T -x 0 . Then, replacing x by x T ,

<!-- formula-not-decoded -->

Let the first term be ˜ ζ = ∇ f ( x 0 ) T ˜ κ + 1 2 ˜ κ T H ( x 0 )˜ κ and the second term be ζ = ∇ f ( x 0 ) T κ + ˜ κ T H ( x 0 ) κ + 1 2 κ T H ( x 0 ) κ + ρ 6 ∥ ˜ κ + κ ∥ 3 . Hence f ( x T ) -f ( x 0 ) ≤ ˜ ζ + ζ . In order to prove the theorem, we require an upper bound on E [ f ( x T ) -f ( x 0 )] .

Now, we introduce two mutually exclusive events C t and ¯ C t so that E [ f ( x T ) -f ( x 0 )] can be written in terms of events C t and ¯ C t as,

<!-- formula-not-decoded -->

Let K 1 = E [ ˜ ζ ] , K 2 = E [ ζ 1 C T ] and K 3 = E [( f ( x T ) -f ( x 0 )) 1 ¯ C T ] -E [ ˜ ζ 1 ¯ C T ] . In the remainder of 575 the proof, we focus on deriving the bounds for individual terms, K 1 , K 2 and K 3 , and then finally put 576 them together to obtain the result of the theorem. 577

˜

## C.1 Bounding K 1 578

Using (20b) from the proof of Lemma 2 in Appendix B.1, we obtain the bound for the term K 1 = E [ ˜ ζ ] 579 as, 580

<!-- formula-not-decoded -->

Since ˜ w τ = 0 , all the terms with E [ ˜ w τ ] will go to zero. Hence we obtain, 581

<!-- formula-not-decoded -->

Let λ 1 , . . . , λ d be the eigenvalues of the Hessian matrix at x 0 , H ( x 0 ) . Now, we simplify similar to 582 Ge et al. [9] as, 583

<!-- formula-not-decoded -->

584

585

586

587

588

589

Note that for the case of very small gradients (as per our initial conditions), |∇ i f ( x 0 ) | 2 ≤ ∥∇ f ( x 0 ) ∥ ≤ ϵ . Therefore, the first and second terms can be made arbitrarily small so that they do not contribute to the order of the equation. Hence, we focus on the third term. We first characterize E [ | ˜ w τ,i | 2 ] as follows. Since the norm of the stochastic noise is bounded as per the assumption A3 , we assume that ˜ g i (˜ x t ) -∇ i ˜ f (˜ x t ) ≤ ˜ q and E [˜ q ] ≤ ˜ σ 2 .

<!-- formula-not-decoded -->

Taking expectation with respect to ˜ q and the uniformly distributed random variable u t +1 and recalling 590 that E [ u t +1 ] = 0 , we set expectation over linear functions of u t +1 to zero. 591

<!-- formula-not-decoded -->

Here, we use E [ u 2 t +1 ] = ( L max -L min ) 2 12 = ˜ O ( L 2 max ) . From (19) in the proof of Lemma 2 (Appendix 592 B.1), ∥ ∥ ∥ ∇ ˜ f (˜ x t ) ∥ ∥ ∥ ≤ 10 ˜ Q ∑ t ( t -1) 2 τ =0 (1 + η c γ o ) τ = ˜ O ( 1 √ L max ) as t ≤ T = ˜ O ( L -1 / 4 max ) . Also, note 593 that ˜ q and u t +1 are independent of each other. As λ min ( H ( x 0 )) = -γ o , 594

<!-- formula-not-decoded -->

where we use the upper bound of E [ | ˜ w τ,i | 2 ] obtained from (39) in (40a). We use the fact that one 595 of the eigenvalues of H ( x 0 ) is -γ o and then upper bound the other eigenvalues by the maximum 596 eigenvalue λ max ( H ( x 0 )) in (40b). 597

598

599

600

Let η c ≤ η max ≤ √ 2 -1 γ ′ where γ ≤ γ o ≤ γ ′ . As ∑ T -1 τ =0 (1 + η c γ o ) 2 τ is a monotonically increasing sequence, we choose the smallest T that satisfies d η 1 / 4 c γ o ≤ ∑ T -1 τ =0 (1+ η c γ o ) 2 τ . Therefore, ∑ T -2 τ =0 (1+ η c γ o ) 2 τ ≤ d η 1 / 4 c γ o . Now,

<!-- formula-not-decoded -->

which follows from our constraints that η c &lt; √ 2 -1 γ ′ and γ o ≤ γ ′ making (1 + η c γ ) 2 ≤ 601 ( 1 + √ 2 -1 γ ′ γ ′ ) 2 ≤ 2 . Further using η c γ o ≤ η 1 / 4 c γ o ≤ √ 2 -1 γ ′ γ ′ &lt; d , 602

<!-- formula-not-decoded -->

Hence the order of T is given by T = O ( log d L 1 / 4 max γ o ) . We hide the dependence on d when we use 603 T = ˜ O ( L -1 / 4 max ) . Using (41) it can be proved that, 604

<!-- formula-not-decoded -->

## C.2 Bounding K 2 and K 3

We define the event C T as, C T = { ∀ t ≤ T, ∥ ˜ κ ∥ ≤ ˜ O ( L 3 / 8 max log 1 L max ) , ∥ κ ∥ ≤ ˜ O ( L 3 / 4 max ) } . From Lemma 2 and Lemma 3 in Appendix B.1 and B.2 respectively, we know that with probability P ( C T ) ≥ 1 -˜ O ( L 7 / 2 max ) , the term ∥ ˜ κ ∥ can be bounded by ˜ O ( L 3 / 8 max log 1 L max ) and ∥ κ ∥ can be bounded by ˜ O ( L 3 / 4 max ) , ∀ t ≤ T = O ( L -1 / 4 max ) .

605

606

607

608

609

Now, to complete the proof of Theorem 2, we need to show that the term K 1 dominates both K 2 and 610 K 3 . Hence, we obtain the bound for the term K 2 as, 611

<!-- formula-not-decoded -->

Finally, we bound the term K 3 as follows. 612

<!-- formula-not-decoded -->

613

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

626

627

where the inequality arises from the boundedness of the function. Comparing the bounds of the terms K 1 , K 2 , and K 3 , we find that K 1 dominates, which completes the proof.

## D Proof of Theorem 3

Theorem 6. (Theorem 3 restated) Consider f satisfying the assumptions A1 -A6 . Let the initial iterate x 0 be δ close to a local minimum x ∗ such that ∥ x 0 -x ∗ ∥ ≤ ˜ O ( √ L max ) &lt; δ . With probability at least 1 -ξ , ∀ t ≤ T where T = ˜ O ( 1 L 2 max log 1 ξ ) ,

<!-- formula-not-decoded -->

Proof. This theorem handles the case when the iterate is close to the local minimum (case B3 ). Weaim to show that the iterate does not leave the neighbourhood of the minimum for t ≤ ˜ O ( 1 L 2 max log 1 ξ ) . By assumption A6 , if x t is δ close to the local minimum x ∗ , the function is locally α - strongly convex. We define event D t = {∀ τ ≤ t, ∥ x τ -x ∗ ∥ ≤ µ √ L max log 1 L max ξ &lt; δ } . Let L max &lt; r log ξ -1 where r &lt; log ξ -1 . It can be seen that D t -1 ⊂ D t . Conditioned on event D t , and using α -strong convexity of f , ( ∇ f ( x t ) -∇ f ( x ∗ )) T ( x t -x ∗ ) 1 D t ≥ α ∥ x t -x ∗ ∥ 2 1 D t . As ∇ f ( x ∗ ) = 0 , it becomes, ∇ f ( x t ) T ( x t -x ∗ ) 1 D t ≥ α ∥ x t -x ∗ ∥ 2 1 D t . We define a filtration S t = s { w 0 , . . . , w t -1 } in order to construct a supermartingale and use the Azuma-Hoeffding inequality where s { . } denotes a sigmaalgebra field. Now, assuming L max &lt; α β 2 ,

<!-- formula-not-decoded -->

We use E [ w t ] = 0 in (42a). We use the β -smoothness and α -convexity assumptions of f in (42b). 628 Now, using w t -1 = η c g ( x t -1 ) -η c ∇ f ( x t -1 ) + u t g ( x t -1 ) , we compute E [ ∥ w t -1 ∥ 2 ] as, 629

<!-- formula-not-decoded -->

As η c = L min + L max 2 , L min = 2 η c -L max . Hence, we write E [ u 2 t ] = ( L max -L min ) 2 12 = 630 4( L max -η c ) 2 12 = L 2 max + η 2 c -2 L max η c 3 &lt; 2 L 2 max 3 -2 L max η c 3 in (43). Using (43) in (42b), 631

<!-- formula-not-decoded -->

We use L max &lt; α β 2 . Let J t = ( 1 + 5 αL max 3 ) -t ( ∥ x t -x ∗ ∥ 2 + L max σ 2 α ) . We prove J t 1 D t -1 is a 632 supermartingale process as follows. 633 634

<!-- formula-not-decoded -->

Hence J t 1 D t -1 is a supermartingale. In order to use the Azuma-Hoeffding inequality, we bound 635 | J t 1 D t -1 -E [ J t 1 D t -1 | S t -1 ] | as, 636 637

<!-- formula-not-decoded -->

where we use (43) in (44) for the term E [ ∥ w t -1 ∥ 2 ] . Now, we compute ∥ w t -1 ∥ using assumption A3 638 as follows. 639

<!-- formula-not-decoded -->

Using (45) in (44) and the bound of the event D t -1 , 640

<!-- formula-not-decoded -->

Hence b t is of the order ˜ O ( µL max log 0 . 5 1 L max ξ ) . By the Azuma Hoeffding inequality, 643

<!-- formula-not-decoded -->

644

645

which leads to,

<!-- formula-not-decoded -->

Hence we can write,

<!-- formula-not-decoded -->

For some constant ˜ b independent of L max and ξ we can write, 646

<!-- formula-not-decoded -->

647

648

By choosing µ &lt; ˜ b ,

<!-- formula-not-decoded -->

Iteratively unrolling the above equation, we obtain P ( ¯ D t ) ≤ t ˜ O ( L 3 max ξ ) . Choosing t = 649 ˜ O ( 1 L 2 max log 1 ξ ) , P ( ¯ D t ) ≤ ˜ O ( L max ξ log 1 ξ ) . As L max &lt; ˜ O ( 1 log 1 ξ ) , P ( ¯ D t ) ≤ ˜ O ( ξ ) . 650

## E Proof using induction 651

In the proof of Lemma 2 in Appendix B.1, we state that (19) can be proved by induction for t ≥ 2 . 652 We restate the equation here and provide the corresponding proof by induction. 653

<!-- formula-not-decoded -->

Recollect from that (15) that ∇ ˜ f (˜ x t ) = ( I -η c H ( x 0 )) ∇ ˜ f (˜ x t -1 ) -H ( x 0 )˜ w t -1 . Taking matrix 654 induced norm on both sides, 655

<!-- formula-not-decoded -->

since, ∥ ∥ ∥ ˜ g (˜ x t ) -∇ ˜ f (˜ x t ) ∥ ∥ ∥ ≤ ˜ Q . Note that ∥ ∥ ∥ ∇ ˜ f (˜ x t ) ∥ ∥ ∥ ≤ ϵ , | u t | ≤ L max and βL max &lt; 1 hold for all 656 t . Therefore, at t = 1 , 657

<!-- formula-not-decoded -->

Now, we prove the hypothesis in (46) for t = 2 . From (47), for an arbitrarily small ϵ , 658

<!-- formula-not-decoded -->

We have shown that the induction hypothesis holds for t = 2 . Now, assuming that it holds for any t , 659 we need to prove that it holds for t +1 . We know from (47), when the hypothesis is assumed to hold 660 for t , 661

<!-- formula-not-decoded -->

If we prove 20 ˜ Q ∑ t ( t -1) 2 +1 τ =0 (1+ η c γ o ) τ ≤ 10 ˜ Q ∑ t ( t +1) 2 τ =0 (1+ η c γ o ) τ , the induction proof is complete. 662 Now, we need to prove 663

<!-- formula-not-decoded -->

Therefore we need to show that, 664

<!-- formula-not-decoded -->

Now, summing up the geometric series S 1 , ∑ t 2 -t 2 +1 τ =0 (1+ η c γ o ) τ = (1+ η c γ o ) t 2 -t 2 +2 -1 η c γ o . Using change 665 of variable in S 2 of (48) as m = τ -( t 2 -t 2 +2 ) , 666

<!-- formula-not-decoded -->

Therefore, we now need to prove, 667

<!-- formula-not-decoded -->

We further prove (49) by induction as follows. For t = 2 , 2(1 + η c γ o ) 3 ≤ (1 + η c γ o ) 4 +1 . Let us 668 assume the following expression holds for time step t . 669

<!-- formula-not-decoded -->

Now, we prove for the time step t +1 ,

<!-- formula-not-decoded -->

where we use t ( t -1) 2 + t = t ( t +1) 2 and apply our assumption (50) in (51). We have proved 2(1 + η c γ o ) t 2 -t 2 +2 ≤ (1 + η c γ o ) t 2 -t 2 + t +1 ≤ (1 + η c γ o ) t 2 -t 2 + t +1 +1 . This concludes our proof of (46).

## F Choice of parameters for other LR schedulers

1. Cosine annealing [21]: There are 3 parameters namely, initial restart interval, a multiplicative factor and minimum learning rate. The authors propose an initial restart interval of 1 , a factor of 2 for subsequent restarts, with a minimum learning rate of 1 e -4 , which we use in our comparisons.
2. Knee [14]: The total number of epochs is divided into those that correspond to the "explore" epochs and "exploit" epochs. During the explore epochs, the learning rate is kept at a constant high value, while from the beginning of the exploit epochs, it is linearly decayed. Weuse the suggested setting of 100 initial explore epochs with a learning rate of 0 . 1 followed by a linear decay for the rest of the epochs.
3. One cycle [25]: We perform the learning rate range test for our networks as suggested by the authors. For the range test, the learning rate is gradually increased during which the training loss explodes. The learning rate at which it explodes is noted and the maximum learning rate (the learning rate at the middle of the triangular cycle) is fixed to be before that. We linearly increase the learning rate for the initial 45% of the total epochs up to the maximum learning rate determined by the range test, followed by a linear decay for the next 45% of the total epochs. We then decay it further up to a divisive factor of 10 for the rest of the epochs, which is the suggested setting. Note that the one cycle LR scheduler relies heavily on regularization parameters like weight decay and momentum.
4. Constant: To compare with a constant learning rate, we choose 0 . 05 for the VGG-16 architecture and 0 . 1 for the remaining architectures as done in our other baselines[24, 21].
5. Multi step: For the multi-step decay scheduler, our choice of the decay rate and time is based on the standard repositories for the architectures. 4 . Specifically, we decay the learning rate by a factor of 10 at the the epochs 100 and 150 for ResNet-110 and ResNet-50. In the case of DenseNet-40-12, we decay by a factor of 10 at the epochs 150 and 225 . For VGG-16, we decay by a factor of 10 every 30 epochs. In the case of WRN, we fix a learning rate of 0 . 2 for the initial 60 epochs, decay it by 0 . 2 2 for the next 60 epochs, and by 0 . 2 3 for the rest of the epochs.

4

ResNet:https://github.com/akamaster/pytorch\_resnet\_cifar10, DenseNet:https://github.com/andreasveit/densenet-pytorch, VGG:https://github.com/chengyangfu/pytorch-vgg-cifar10, WRN:https://github.com/meliketoy/wide-resnet.pytorch

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

Figure 4: Reconstruction error for online tensor decomposition

<!-- image -->

## G Online tensor decomposition

We follow the experimental setup in [9], where their proposed projected noisy gradient descent is 702 applied to orthogonal tensor decomposition. A brief description of the online tensor decomposition 703 problem is given below. 704

705

Consider a tensor T which has an orthogonal decomposition,

<!-- formula-not-decoded -->

where a i 's are orthonormal vectors. The goal of performing the tensor decomposition is to find the 706 orthonormal components, given the tensor. The objective function is defined to reduce the correlation 707 between the components: 708

̸

<!-- formula-not-decoded -->

We plot the normalized reconstruction error, ∥ ∥ ∥ T -∑ d i =1 u ⊗ 4 i ∥ ∥ ∥ 2 F / ∥ T ∥ 2 F in Figure 4, where ∥ . ∥ F 709 denotes the Frobenius norm. We tune the learning rate parameters L min and L max to 0 . 007 and 710 0 . 01 respectively to obtain the convergence plot with PLRS. We compare against the plot in Figure 711 1.a of [9]. We note that the proposed Uniform LR produces faster and smoother convergence when 712 compared to the unit sphere noise proposed in the Noisy SGD algorithm. As mentioned in [9], the 713 plot may vary depending on the instance of initialization; however, it converges consistently across 714 all runs. 715

Additionally, we implemented stochastic gradient descent with additive noise in the neural network 716 setting. However, its performance was suboptimal even with extensive tuning of hyperparameters. 717

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

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We give a detailed account of our contributions in Section 1.2.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations of our work in Section 5.4.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We give detailed and correct proofs for all of our theoretical results in the Appendices, with a short sketch/description in the main paper.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide all the details to reproduce the empirical results of the paper in Section 5.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We provide our code in the supplemental material.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We provide exact values of the hyperparameters and details of the experiments in Section 5. Additional details on the hyperparameters of the benchmarks are provided in the Appendix F.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report the mean and standard deviation of test accuracy over three independent runs for each experiment.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

765

766

767

768

769

770

771

772

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

Justification: We provide details about the computational setup in Section 5.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the impact of our work in Section 5.4.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: In Section 5, we cite the original source whenever a neural network architecture or dataset is referenced.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We provide our code with documentation in the supplemental material and also credit the codes which we use.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

811

812

Justification: We do not use LLMs as a core component of this work.
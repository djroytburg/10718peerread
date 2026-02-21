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

## Curse of Slicing: Why Sliced Mutual Information is a Deceptive Measure of Statistical Dependence

## Anonymous Author(s)

Affilation, Address anon.email@example.org

## Abstract

Sliced  Mutual  Information  (SMI)  is  widely  used  as  a  scalable  alternative  to mutual information for measuring non-linear statistical dependence. Despite its advantages, such as faster convergence, robustness to high dimensionality, and nullification  only  under  statistical  independence,  we  demonstrate  that  SMI  is highly  susceptible  to  data  manipulation  and  exhibits  counterintuitive  behavior. Through  extensive  benchmarking  and  theoretical  analysis,  we  show  that  SMI saturates easily, fails to detect increases in statistical dependence (even under linear transformations  designed  to  enhance  the  extraction  of  information),  prioritizes redundancy over informative content, and in some cases, performs worse than simpler dependence measures like the correlation coefficient.

Ability to capture complex statistical dependencies

<!-- image -->

## 1 Introduction

- Mutual information (MI) is a fundamental and invariant measure of nonlinear statistical dependence 21
- between two random vectors, defined as the Kullback-Leibler divergence between the joint distrib-

ution and the product of marginals [1]: 22 23

<!-- formula-not-decoded -->

- Due  to  several  outstanding  properties,  such  as  nullification  only  under  statistical  independence, 24
- invariance to invertible transformations, and ability to capture non-linear dependencies, MI is used 25
- extensively for theoretical analysis of overfitting [2], [3], hypothesis testing [4], feature selection [5], [6], [7], representation learning [8], [9], [10], [11], [12], [13], and studying the mechanisms behind generalization in deep neural networks (DNNs) [14], [15], [16], [17]. 26 27 28
- In practical scenarios, ℙ𝑋,𝑌 and ℙ𝑋 ⊗ ℙ 𝑌 are unknown, requiring MI to be estimated from finite 29
- samples. Despite all the aforementioned merits, this reliance on empirical estimates leads to the curse 30

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

72

73

74

75

76

of dimensionality: the sample complexity of MI grows exponentially with the number of dimensions [18],  [19].  A  common strategy to mitigate this issue is to use alternative measures of statistical dependence that are more stable in high dimensions. However, such measures usually offer only a fraction of MI capabilities. Therefore, it is crucial to maintain a balance between robustness to the curse of dimensionality and the ability to detect complex dependency structures.

To  strike  this  balance,  popular  techniques  often  retain  MI  as  a  backbone  statistical  measure but employ dimensionality reduction before estimation. While some studies explore sophisticated nonlinear compression methods [17], [20], others favor more scalable linear projection approaches [21], [22], [23], [24], [25]. Among the latter group, the Sliced Mutual Information (SMI) [22], [23] stands out, leveraging random projections to cover all directions uniformly:

<!-- formula-not-decoded -->

Uniform slicing allows SMI to maintain some crucial properties of MI (e.g., being zero if and only if 𝑋 and 𝑌 are independent), while remaining completely free from additional optimization problems (e.g., from finding optimal projections, as in [24], [25]). Combined with fast convergence rates, this has established SMI as a scalable alternative to MI. Consequently, it has been widely adopted for studying DNNs [26], [27], [28], [29], [30], deriving generalization bounds [31], independence testing [32] and auditing differential privacy [33]. It was also proposed to use SMI for feature selection [22] and preventing mode collapse in generative models [23].

Despite its popularity, the research community has largely overlooked potential shortcomings of SMI. Some studies prematurely attribute their results to underlying phenomena without rigorously investigating  whether  they  stem  from  artifacts  introduced  by  random  projections.  Furthermore, existing works fail to comprehensively address issues related to random slicing, focusing primarily on suboptimality of random projections for information preservation [24], [25].

Contribution. In this article, we address this gap by systematically analyzing SMI across diverse settings, demonstrating that it frequently exhibits counterintuitive behavior and fails to accurately capture statistical dependence dynamics. Our key contributions are:

1. Saturation and Sensitivity Analysis. Through theoretical analysis and extensive benchmarking, we show that SMI saturates prematurely, even for low-dimensional synthetic problems, and fails to detect significant increases in statistical dependence.
2. Redundancy Bias. We  refute  the  prevailing  assumption  that  SMI  favors  linearly  extractable information by constructing an explicit example where introducing such structure increases MI and even linear correlation, but decreases SMI. In fact, we show that SMI prioritizes information redundancy over information content. We argue that this bias can lead to catastrophic failures in some applications, e.g. collapses in representation learning.
3. Curse of Dimensionality. We revisit the dynamics of SMI for increasing dimensionality and argue that SMI is, in fact, cursed, with the curse of dimensionality manifesting itself not through sample complexity, but via asymptotic decay to zero in high-dimensional regimes due to diminishing redundancy.
4. Reestablishing the Trade-off. Finally, we discuss to which extent the aforementioned problems can be solved by using non-uniform/non-random slicing strategies, and how they affect the tradeoff between scalability and utility of different measures of statistical dependence.

Our paper is structured as follows. In Section 2, we provide the mathematical background that is necessary for our analysis. In Section  3, we discuss previous findings which are related to the research topic of this work. Section 4 consists of our main theoretical results, with the complete proofs being provided in Section B. In Section 5, we employ synthetic benchmarks to show the disconnection between dynamics of MI and SMI. Section 6 illustrates that tasks related to SMI maximization may yield degenerate solutions, contrary to MI maximization. Finally, we discuss our results in Section 7.

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

## 2 Preliminaries

Elements of Information Theory. Let (Ω, ℱ, ℙ) be a probability space with sample space Ω , 𝜎 -algebra ℱ , and probability measure ℙ defined on ℱ . Consider random vectors 𝑋 : Ω → ℝ 𝑑𝑥 and 𝑌 : Ω → ℝ 𝑑𝑦 with joint distribution ℙ𝑋,𝑌 and marginals ℙ𝑋 and ℙ𝑌 , respectively. Wherever it is needed, we assume the relevant Radon-Nikodym derivatives exist. For any probability measure ℚ ≪ ℙ , the Kullback-Leibler (KL) divergence is D𝖪𝖫 (ℚ ‖ ℙ) = 𝔼 ℚ [log dℚ dℙ ] , which is non-negative and vanishes if and only if (iff) ℙ = ℚ . The mutual information (MI) between 𝑋 and 𝑌 quantifies the divergence between the joint distribution and the product of marginals:

<!-- formula-not-decoded -->

When ℙ𝑋 admits a probability density function (PDF) 𝑝(𝑋) with respect to (w.r.t.) the Lebesgue measure, the differential entropy is defined as 𝗁(𝑋) = -𝔼[log𝑝(𝑋)] ,  where log( ⋅ ) denotes the natural  logarithm.  Likewise,  the  joint  entropy 𝗁(𝑋,𝑌 ) is  defined  via  the  joint  density 𝑝(𝑋,𝑌 ) , and conditional entropy is 𝗁(𝑋 | 𝑌 ) = - 𝔼[log 𝑝(𝑋 | 𝑌 )] = - 𝔼 𝑌 [𝔼 𝑋 | 𝑌 log 𝑝(𝑋 | 𝑌 )] . Under the existence of PDFs, MI satisfies the identities

<!-- formula-not-decoded -->

In this work, we denote by 𝜇M the normalized Haar (uniform) probability measure on a compact manifold M , i.e., the unique bi-invariant measure satisfying 𝜇M(M) = 1 . Hence, to sample uniformly from specific spaces we write W∼𝜇O(𝑑),𝜃 ∼ 𝜇 𝕊 𝑑-1, A ∼ 𝜇 St(𝑘,𝑑) , indicating draws from the Haar measures on orthogonal group O(𝑑) = {Q ∈ ℝ 𝑑×𝑑 : Q 𝖳 Q = QQ 𝖳 = I} ,  the  unit sphere 𝕊 𝑑-1 = {𝑋 ∈ ℝ 𝑑 : ‖𝑋‖ 2 = 1} , and the Stiefel manifold St(𝑘, 𝑑) = {Q ∈ ℝ 𝑑×𝑘 : Q 𝖳 Q = I} , respectively.

Sliced Mutual Information. To mitigate the curse of dimensionality, one may average MI over all 𝑘 -dimensional projections. The 𝑘 -sliced mutual information ( 𝑘 -SMI) [23] between 𝑋 and 𝑌 is defined as

<!-- formula-not-decoded -->

which can be efficiently estimated. Setting 𝑘 = 1 recovers the standard sliced mutual information (1).

## 3  Background

Merits of SMI are straightforward and have been investigated thoroughly in [22], [23]. We remind the reader of the two most important of them:

1. Scalability (i.e., fast convergence in high dimensions), enabled by low-dimensional projections.
2. Nullification Property (i.e., 𝖲𝖨 𝑘 (𝑋;𝑌 ) = 0 iff 𝑋 and 𝑌 projections being random and independent.
3. are independent), which stems from the

In  contrast,  demerits of SMI are not very obvious and not well-covered in the literature. In this section, we recapitulate and analyze previous works which address the shortcomings of SMI. To facilitate the analysis, we divide them into three main categories.

Suboptimality of random slicing. In [24] and [25], it is argued that a uniform slicing strategy can produce suboptimal projections, impairing SMI's ability to capture dependencies in the presence of noisy or non-informative components. To address this issue, [24] proposed max-sliced MI (mSMI), which selects non-random projectors that maximize the MI between projected representations. This approach is also claimed to improve interpretability and convergence rates.

However, deterministic slicing may overlook dependencies captured by non-optimal components. To mitigate this, [25] extends the max-sliced approach by optimizing SMI over probability distributions of projectors, with regularization to maintain slice diversity. While the authors emphasize that optimization should occur over joint distributions, their motivation primarily addresses the issue of non-optimal marginal distributions of 𝜃 and 𝜙 - specifically, the presence of non-informative

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

components in 𝑋 and 𝑌 . We contend that this represents only a partial understanding of the problem, as many SMI artifacts arise from other factors. Needless to say that optimization over probability distributions is also a heavy burden, which does not align with the slicing philosophy.

Data Processing Inequality violation. A fundamental property of MI is that it cannot be increased by deterministic processing or, more generally, by Markov kernels. Furthermore, MI is preserved under invertible transformations. This is formalized by the data processing inequality (DPI).

Theorem 3.1. (Theorem 3.7 in [1]) For a Markov chain 𝑋 → 𝑌 → 𝑍 , 𝖨(𝑋; 𝑌 ) ≥ 𝖨(𝑋; 𝑍) . Additionally, if 𝑍 = 𝑓(𝑌 ) where 𝑓 is measurably invertible, then equality holds.

In contrast to MI, SMI violates the DPI (see Section 3.2 in [22] for an example). While the intuition behind DPI is clear (raw data already contains full information, and processing can only destroy it), the implications of DPI violation are less straightforward.

Existing works suggest that SMI's violation of DPI can reflect a preference for linearly extractable features, framing this as a useful property that aligns with the informal understanding of 'practically available' (i.e., easily accessible) information [22], [26], [30]. However, this interpretation can be misleading if the factors behind SMI increases are misidentified. Our analysis reveals that this is indeed the case, as SMI exhibits more inherent biases than previously recognized.

Asymptotics in high-dimensional regime. Convergence analysis suggests that the sample complexity of SMI estimation is far less sensitive to data dimensionality compared to that of MI. In fact, it has been argued that the estimation error may even decrease with dimensionality in some cases (see Remark 4 in [23]). However, an analysis of SMI itself reveals that this behavior may result from the fact that SMI can decrease as dimensionality grows. Specifically, Theorem 3 in [23] provides an asymptotic expression (as 𝑑 → ∞ ) for SMI in the case of jointly normal 𝑋 and 𝑌 , which decays hyperbolically with 𝑑 under some circumstances.

To  date,  no  explanation  for  this  phenomenon  has  been  provided  in  the  literature.  We  therefore elaborate on this finding by deriving non-asymptotic expressions, along with experimental results for non-Gaussian data, which reveal further nuances behind the decay.

## 4 Theoretical analysis

We start our analysis with considering a simple example, which (a) admits closed-form expression for SMI and (b) is capable of illustrating severe problems of the quantity in question.

Lemma 4.1. Consider the following pair of jointly Gaussian 𝑑 -dimensional random vectors:

<!-- formula-not-decoded -->

In this setup, MI and SMI can be calculated analytically:

<!-- formula-not-decoded -->

where 3𝐹 2 is the generalized hypergeometric function . Additionally, the following limits hold:

<!-- formula-not-decoded -->

with 𝜓 being the digamma function .

Note that while MI correctly captures the growing statistical dependence as 𝑑 → ∞ (since additional components contribute shared information), SMI drops to zero, exposing a fundamental problem. This issue was briefly noted in [23], but only through providing an asymptotic expression without further discussion. We interpret this behavior as a distinct manifestation of the curse of dimensionality : as 𝑑 grows, SMI uniformly decays to zero and becomes ineffective for statistical analysis. 151 152 153 154 155

<!-- formula-not-decoded -->

<!-- image -->

173

174

175

Figure 2: Saturation of 𝖲𝖨(𝑋; 𝑌 ) as function of 𝖨(𝑋; 𝑌 )/𝑑 for the example from Lemma 4.1, nonnormalized (left) and normalized (right) versions. Note that the problem becomes more prominent in higher dimensions, both because of lower plateau and faster saturation.

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

The second pair of limits reveals another critical flaw of SMI. When 𝜌 2 →1 , the 𝑋 -𝑌 relationship becomes deterministic - a property MI reflects successfully. In stark contrast, SMI remains bounded by a dimension-dependent factor that decays hyperbolically. Furthermore, plotting SMI against MI shows this bound is reached prematurely, demonstrating SMI's rapid saturation with increasing dependence (Figure 2).  In  this  saturated  regime,  SMI  becomes  effectively  insensitive  to  further growth in shared information. Moreover, this renders estimates of SMI for different dimensionalities fundamentally incomparable, as they are theoretically bounded by factors depending on 𝑑 .

These phenomena can not be explained by suboptimality of individual projections. In fact, each individual  projection  is  optimal,  as 𝖨(𝜃 𝖳 𝑋; 𝑌 ) does  not  depend  on 𝜃 in  this  particular  example. The proof of Lemma 4.1 suggests that the problem arises from the majority of pairs of projectors being suboptimal, yielding near-independent 𝜃 𝖳 𝑋 and 𝜙 𝖳 𝑌 in the most outcomes, even for 𝑑 = 2 . Although similar analysis for 𝑘 -SMI is extremely challenging, we argue that the problems in question prevail even when employing 𝑘 -rank projectors.

Proposition 4.2. Under the setup of Lemma 4.1, 𝑘 -SMI has the following integral representation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark. 4.3. As the dimension 𝑑 grows, the term (⋆) asymptotically concentrates the eigenvalues 𝜆 𝑖 near zero, leading to the decay of 𝖲𝖨 𝑘 to zero.

We  argue  that  the  limitations  we  uncovered  can  be  attributed  to  a  strong  bias  of  SMI  toward information redundancy . That is, SMI favors repetition of information across different axes, and suffers from the curse of dimensionality if 𝑋 and 𝑌 have high entropy. The following proposition and remark present a simple example to clarify this bias.

Proposition 4.4. Let 𝑋 and 𝑌 be 𝑑 𝑥 , 𝑑 𝑦 -dimensional random vectors correspondingly, with 𝑑 𝑥 , 𝑑 𝑦 &lt; 𝑘 . Let A ∈ ℝ 𝑚𝑥×𝑑𝑥 and B ∈ ℝ 𝑚𝑦×𝑑𝑦 be matrices of ranks 𝑑 𝑥 , 𝑑 𝑦 . Then 𝖲𝖨 𝑘 (A𝑋;B𝑌 ) = 𝖨(𝑋; 𝑌 ) .

Corollary 4.5. Consider the following pair of jointly Gaussian 𝑑 -dimensional random vectors:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark. 4.6. Applying 𝟏 ⋅ 𝑒 𝖳 1 to the random vectors from Lemma 4.1 individually yields the example from Corollary 4.5. Therefore, this linear transform increases SMI despite decreasing MI.

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

239

240

241

242

243

244

## 4.1 Extension to optimal slicing

Although our work primarily focuses on conventional (average) sliced mutual information (SMI), as it is the most widely used variant, we also provide some intuition regarding the limitations of its 'optimal' counterparts: max-sliced MI (mSMI) [24] and optimal-sliced MI (oSMI) [25]. Since mSMI is a special case of oSMI without regularization constraints, we restrict our discussion to mSMI, though our reasoning extends to oSMI as well. The 𝑘 -mSMI is defined as:

<!-- formula-not-decoded -->

To highlight the shortcomings of linear compression, we revisit a Gaussian example. The following proposition demonstrates that even in this simple setting, mSMI captures only a subset of dependencies and can exhibit opposite trends to MI. This occurs, for instance, when dependencies become more evenly distributed across components, which again returns us to the redundancy bias .

Proposition 4.7. (Proposition 2 in [24]) Let (𝑋,𝑌 ) ∼ 𝒩(𝜇, Σ) ,  with marginal covariances Σ𝑋 , Σ𝑌 and cross-covariance Σ𝑋𝑌 . Suppose the matrix Σ - 1 2 𝑋 Σ𝑋𝑌 Σ - 1 2 𝑌 exists, and let {𝜌 𝑖 } 𝑑 𝑖=1 denote its singular values in descending order, where 𝑑 = min(𝑑 𝑥 , 𝑑 𝑦 ) . Then

<!-- formula-not-decoded -->

## 5 Synthetic Experiments

To complement the theoretical analysis from the previous section and address complex, non-Gaussian cases, we conduct an extensive benchmarking of SMI using synthetic tests from [34], based on the works of [35], [36]. This benchmark suite is used to evaluate MI estimators. However, we do not assess whether SMI estimates converge to ground-truth MI values. SMI is a distinct measure of statistical dependance , and should not be viewed as an approximation of MI. Instead, our analysis focuses on the relationship between the two measures: since MI captures the true degree of statistical dependence, opposing trends in MI and SMI reveal problems with the latter quantity.

For  the  experiments,  we  use correlated  normal , correlated  uniform , smoothed  uniform and loggamma-exponential distributions, for which the ground-truth value of MI is available. To increase the  dimensionality,  we  use  independent  components  with  equally  distributed  per-component  MI. These setups will be referred to as 'randomized' and 'non-randomized' correpsondingly. For each distribution, we vary both the data dimensionality ( 𝑑 ) and the projection dimensionality ( 𝑘 &lt; 𝑑 ).

To estimate MI between projections, we use the KSG estimator [35] with the number of neighbors fixed at 1 .  For  each  configuration, we conduct 10 independent runs with different random seeds to compute means and standard deviations. Our experiments use 10 4 samples for (𝑋,𝑌 ) and 128 samples for (Θ, Φ) .

To experimentally verify saturation, we plot SMI against MI normalized by dimensionality 𝑑 in Figure 3.  The  plots  clearly  show  that  SMI  reaches  a  plateau  relatively  early  for  all  the  featured distributions. The results for the normal distribution also align well with those from Lemma 4.1. We further confirm the saturation of 𝑘 -SMI for 𝑘 ∈ {2, 3} experimentally in Section C. Finally, we plot the saturated values against 𝑑 on a log-log scale, demonstrating that the 1/𝑑 trend from Lemma 4.1 also holds for non-Gaussian distributions.

## 6 SMI for InfoMax-like tasks

Since mutual information is interpretable and captures non-linear dependencies, it is widely used as a training objective. Many applications involve maximizing MI (InfoMax) for feature selection [5], [6], [7] and self-supervised representation learning [8], [9], [10], [11], [12], [13]. However, due to the curse of dimensionality, alternative objectives have been proposed, with some works using sliced mutual information maximization for feature extraction [22] and disentanglement in InfoGAN [23].

In this section, we argue that SMI is not a suitable alternative to MI for InfoMax tasks. Since SMI exhibits a strong preference for redundancy, SMI maximization may lead to collapsed (high-redundancy) solutions. We demonstrate this through two experiments. Firstly, we revisit the Gaussian noisy channel to demonstrate that SMI favors linear mappings which decrease robustness to noise. Then, we consider a self-supervised representation learning task and show that using SMI immediately leads to collapsed representations. 245 246 247 248 249 250

<!-- image -->

277

278

279

Figure 3: Results of synthetic experiments with different distributions. We report mean values and standard deviations computed across 10 runs, with 10 4 samples used for MI estimation and 128 for averaging across projections.

<!-- image -->

290

291

292

Figure 4: Decaying trends of 𝑘 -SMI for correlated normal (corr. 𝒩 ), correlated uniform (corr. U ), smoothed uniform (sm. U ) and log-gamma-exponential (LGE). We plot saturated values of 𝑘 -SMI against data dimensionality 𝑑 . Log scale is used to illustrate the 1/𝑑 trend predicted in Lemma 4.1.

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

320

321

322

323

324

## 6.1 Gaussian Channel

Let 𝑋 be  a  zero-mean 𝑑 -dimensional  random  vector,  and  let 𝑍 ∼ 𝒩(0,𝜎I) be  an  independent noise. Additive white noise Gaussian (AWGN) channel is defined as 𝑋 → 𝑋 + 𝑍 . Maximization of 𝖨(𝑋; 𝑋 + 𝑍) w.r.t. the distribution of 𝑋 is a classical information transmission problem, which arises in many fields under the Gaussian noise assumption. Given energy constraints, it admits an analytical solution [37]:

<!-- formula-not-decoded -->

It is somewhat intuitive that unit covariance matrix allows for more information to be transmitted, as all the components of 𝑋 are utilized to full extent. However, due to the redundancy bias, SMI prefers less robust distributions. To demonstrate this, we consider two linear normalization mappings which impose energy constraints on a vector 𝑋 with zero mean and covariance Σ :

1. Whitening : Σ -1/2 𝑋 ;
2. Standardization : D -1/2 𝑋 , where D = diag(Σ) .

We  conduct  numerical  experiments  for 𝜎 = 0.1 , 𝑋 ′ ∼ A ⋅ U([-1; 1] 5 ) and 𝑋 ″ ∼ A ⋅ 𝒩(0, I 5 ) , where A = 10 -2 ⋅ I + 𝟏 ⋅ 𝟏 𝖳 is  an  ill-conditioned  matrix.  We  employ  the  same  estimators  and hyperparameters as in Section 5. The results are presented in Table 1.

Table 1: Results for additive white Gaussian noise channel ( 𝜎 = 0.1 ), mean and std for 10 runs.

|     | MI         | MI         | SMI        | SMI        | 2 -SMI     | 2 -SMI     |
|-----|------------|------------|------------|------------|------------|------------|
|     | Σ -1/2     | D -1/2     | Σ -1/2     | D -1/2     | Σ -1/2     | D -1/2     |
| 𝑋 ′ | 7.48 ±0.01 | 3.04 ±0.01 | 0.17 ±0.02 | 1.82 ±0.04 | 0.96 ±0.04 | 2.46 ±0.03 |
| 𝑋 ″ | 7.49 ±0.02 | 3.04 ±0.01 | 0.14 ±0.02 | 1.83 ±0.04 | 0.82 ±0.05 | 2.49 ±0.05 |

## 6.2 Representation Learning

To further demonstrate SMI's sensitivity to information redundancy, we examine its performance in learning compressed representations through mutual information maximization ( Deep InfoMax ) [8]. This approach is known to be equivalent to many popular contrastive self-supervised learning methods [13].

In Deep InfoMax, an encoder network 𝑓 is trained to maximize a lower bound on 𝖨(𝑋; 𝑓(𝑋)) , where 𝑋 represents input data and 𝑓(𝑋) its compressed representation. This method is theoretically sound, as maximizing MI ensures the most informative embeddings under the latent space dimensionality

Figure 5: Visualizations of embeddings from the representation learning experiments, with points colored by class. Note that mutual information maximization (left) produces clustered low-redundancy representations, while SMI maximization results in immediate (after 10 epochs) collapse.

<!-- image -->

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

369

370

constraint. For our study, we replace MI with SMI in this framework. This substitution is straightforward since both MI and SMI admit Donsker-Varadhan variational lower bounds [38]:

<!-- formula-not-decoded -->

where 𝑇 is a critic function, which is also approximated in practice by a neural network. For detailed derivations of these bounds, we refer the reader to [39] (MI) and [22], [23] (SMI).

We strictly follow the experimental protocol from [13]. In particular, we use MNIST handwritten digits dataset [40], employ InfoNCE loss [41] to approximate (5), use convolutional network for 𝑓 and fully-connected network for 𝑇 . Latent space dimensionality is fixed at 𝑑 = 2 for visualization purposes.  Small  Gaussian  noise  is  added  to  the  outlet  of  the  encoder  to  combat  representation collapse [13]. More details are provided in Section D. We focus on this simple setup because our objective is to show that SMI produces degenerate results even in elementary tasks, making more complex configurations unnecessary for this demonstration.

Results are presented in Figure 5. As our theory predicts, maximization of SMI immediately leads to collapsed representations, while conventional InfoMax yields embeddings with low or even zero redundancy (components are close to 𝒩(0,I) ). This behavior is consistent across different runs.

## 7 Discussion

Results. Sliced mutual information (SMI) has been proposed as a scalable alternative to Shannon's mutual information. While SMI enables efficient computation in high-dimensional settings and satisfies the nullification property, our findings reveal critical deficiencies that undermine its reliability for feature extraction and related tasks.

We demonstrate that SMI saturates rapidly, failing to capture variations in statistical dependence. This makes it difficult to distinguish between intrinsic SMI fluctuations and genuine changes in dependence structure. Furthermore, we invalidate the common hypothesis that SMI favors linear features  through  a  counterexample  where  even  correlation  coefficients  reflect  dependence  more faithfully than SMI, which exhibits inverted behavior.

In high-dimensional spaces, SMI decays with increasing dimensionality, contrary to MI's monotonic behavior. This is established analytically for Gaussian cases and validated empirically across diverse synthetic experiments. Consequently, SMI variations may reflect redundancy, dependence changes, or high-dimensional artifacts without a principled way to disentangle these factors.

Impact. Thanks to fast convergence rates and the absence of additional optimization problems, SMI has been widely applied across various fields of statistics and machine learning. Given our findings, it is therefore crucial to recognize how the inherent biases of SMI affect practical applications.

The works [22] and [23] propose using SMI in a Deep InfoMax setting. However, we demonstrate that maximizing SMI can lead to collapsed solutions due to redundancy bias. Meanwhile, [26], [27], [28], [30] study deep neural networks by measuring SMI between intermediate layers. Yet, as our analysis reveals, changes in SMI do not always reflect true shifts in statistical dependence; they may instead result from differences in layer dimensionality, redundancy in intermediate representations, low sensitivity in saturated regimes, or other factors. Finally, [33] suggests using SMI for independence testing in differential privacy tasks. We contend that this approach poses critical issues, as SMI estimates  can  become  statistically  indistinguishable  from  zero  in  high-dimensional  or  lowredundancy settings.

Limitations. While  we  support  our  claims  with  both  theoretical  analysis  and  experimental evidence, we were able to derive analytical expressions for the Gaussian case only. Furthermore, our synthetic tests do not feature complex, highly non-linear distributions (such as structured image data used in [17]). Nevertheless, we demonstrate that our findings are more than sufficient to expose fundamental limitations of SMI, and to support all the claims we made.

## References 371

- [1] Y. Polyanskiy and Y. Wu, Information Theory: From Coding to Learning . Cambridge University Press, 2024. [Online].  Available: https://books.google.ru/books?id=CySo0AEACAAJ 372 373
- [2] A.  Asadi,  E.  Abbe,  and  S.  Verdu,  'Chaining  Mutual  Information  and  Tightening  Generalization Bounds,' in Advances in Neural Information Processing Systems , S. Bengio, H. Wallach, H. Larochelle, K.  Grauman,  N.  Cesa-Bianchi,  and  R.  Garnett,  Eds.,  Curran  Associates,  Inc.,    2018,  p.  .  [Online]. Available:  https://proceedings.neurips.cc/paper\_files/paper/2018/file/8d7628dd7a710c8638dbd22d4421 ee46-Paper.pdf 374 375 376 377 378
- [3] J. Negrea, M. Haghifam, G. K. Dziugaite, A. Khisti, and D. M. Roy, 'Information-Theoretic Generalization Bounds for SGLD via Data-Dependent Estimates,' in Advances in Neural Information Processing Systems , H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, Eds., Curran Associates, Inc.,  2019, p. . [Online].  Available: https://proceedings.neurips.cc/paper\_files/paper/2019/ file/05ae14d7ae387b93370d142d82220f1b-Paper.pdf 379 380 381 382 383
- [4] B. Duong and T. Nguyen, 'Conditional Independence Testing via Latent Representation Learning,' in 2022 IEEE International Conference on Data Mining (ICDM) , Los Alamitos, CA, USA: IEEE Computer Society, Dec. 2022, pp. 121-130. doi: 10.1109/ICDM54844.2022.00022. 384 385 386
- [5] S. Yang and J. Gu, 'Feature selection based on mutual information and redundancy-synergy coefficient,' J. Zhejiang Univ. Sci. , vol. 5, no. 11, pp. 1382-1391, Nov. 2004. 387 388
- [6] N. Kwak and C.-H. Choi, 'Input feature selection by mutual information based on Parzen window,' IEEE Transactions on Pattern Analysis and Machine Intelligence , vol. 24, no. 12, pp. 1667-1671, 2002, doi: 10.1109/TPAMI.2002.1114861. 389 390 391
- [7] M. A. Sulaiman and J. Labadin, 'Feature selection based on mutual information,' in 2015 9th International Conference on IT in Asia (CITA) ,  2015, pp. 1-6. doi: 10.1109/CITA.2015.7349827. 392 393
- [8] R. D. Hjelm et al. , 'Learning deep representations by mutual information estimation and maximization,' in International Conference on Learning Representations ,  2019. [Online].  Available: https://openreview. net/forum?id=Bklr3j0cKX 394 395 396
- [9] P. Bachman, R. D. Hjelm, and W. Buchwalter, 'Learning Representations by Maximizing Mutual Information Across Views,' in Advances in Neural Information Processing Systems , H. Wallach, H. Larochelle, A.  Beygelzimer,  F.  d'Alché-Buc,  E.  Fox,  and  R.  Garnett,  Eds.,  Curran  Associates,  Inc.,    2019,  p.  . [Online].  Available: https://proceedings.neurips.cc/paper\_files/paper/2019/file/ddf354219aac374f1d40b 7e760ee5bb7-Paper.pdf 397 398 399 400 401
- [10] P. Veličković, W. Fedus, W. L. Hamilton, P. Liò, Y. Bengio, and R. D. Hjelm, 'Deep Graph Infomax,' in International Conference on Learning Representations ,  2019. [Online].  Available: https://openreview. net/forum?id=rklz9iAcKQ 402 403 404
- [11] M. Tschannen, J. Djolonga, P. K. Rubenstein, S. Gelly, and M. Lucic, 'On Mutual Information Maximization for Representation Learning,' in International Conference on Learning Representations ,  2020. [Online].  Available: https://openreview.net/forum?id=rkxoh24FPH 405 406 407
- [12] X. Yu, 'Leveraging Superfluous Information in Contrastive Representation Learning.' [Online]. Available: https://arxiv.org/abs/2408.10292 408 409
- [13] I. Butakov, A. Semenenko, A. Tolmachev, A. Gladkov, M. Munkhoeva, and A. Frolov, 'Efficient Distribution Matching of Representations via Noise-Injected Deep InfoMax,' in The Thirteenth International Conference on Learning Representations ,  2025. [Online].  Available: https://openreview.net/forum?id= mAmCdASmJ5 410 411 412 413
- [14] N. Tishby and N. Zaslavsky, 'Deep learning and the information bottleneck principle,' 2015 IEEE Information Theory Workshop (ITW) , pp. 1-5, 2015. 414 415
- [15] R. Shwartz-Ziv and N. Tishby, 'Opening the Black Box of Deep Neural Networks via Information.' 2017. 416

417

418

419

420

Z. Goldfeld et al.

, 'Estimating Information Flow in Deep Neural Networks,' in

International Conference on Machine Learning

Proceedings of the 36th

, K. Chaudhuri and R. Salakhutdinov, Eds., in Proceedings of  Machine  Learning  Research,  vol.  97.  PMLR,    2019,  pp.  2299-2308.  [Online].    Available:  https://

proceedings.mlr.press/v97/goldfeld19a.html

- [17] I.  Butakov, A. Tolmachev, S. Malanchuk, A. Neopryatnaya, A. Frolov, and K. Andreev, 'Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression,' in The Twelfth International 421 422

[16]

- Conference on Learning Representations ,  2024. [Online].  Available: https://openreview.net/forum?id= huGECz8dPp 423 424

425

426

427

- [18] Z. Goldfeld, K. Greenewald, J. Niles-Weed, and Y. Polyanskiy, 'Convergence of Smoothed Empirical Measures With Applications to Entropy Estimation,' IEEE Transactions on Information Theory , vol. 66, no. 7, pp. 4368-4391, 2020, doi: 10.1109/TIT.2020.2975480.

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

466

467

468

- [19] D.  McAllester  and  K.  Stratos,  'Formal  Limitations  on  the  Measurement  of  Mutual  Information,'  in Proceedings  of  the  Twenty  Third  International  Conference  on  Artificial  Intelligence  and  Statistics ,  S. Chiappa and R. Calandra, Eds., in Proceedings of Machine Learning Research, vol. 108. PMLR,  2020, pp. 875-884. [Online].  Available: https://proceedings.mlr.press/v108/mcallester20a.html
- [20] G. Gowri, X. Lun, A. M. Klein, and P. Yin, 'Approximating mutual information of high-dimensional variables using learned representations,' in The Thirty-eighth Annual Conference on Neural Information Processing Systems ,  2024. [Online].  Available: https://openreview.net/forum?id=HN05DQxyLl
- [21] K.  H.  Greenewald,  B.  Kingsbury,  and  Y.  Yu,  'High-Dimensional  Smoothed  Entropy  Estimation  via Dimensionality Reduction,' in IEEE International Symposium on Information Theory, ISIT 2023, Taipei, Taiwan, June 25-30, 2023 , IEEE,  2023, pp. 2613-2618. doi: 10.1109/ISIT54713.2023.10206641.
- [22] Z. Goldfeld and K. Greenewald, 'Sliced Mutual Information: A Scalable Measure of Statistical Dependence,' in Advances in Neural Information Processing Systems , A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, Eds.,  2021. [Online].  Available: https://openreview.net/forum?id=27qon5Ut4PSl
- [23] Z. Goldfeld, K. Greenewald, T. Nuradha, and G. Reeves, '$k$-Sliced Mutual Information: A Quantitative Study of Scalability with Dimension,' in Advances in Neural Information Processing Systems , A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, Eds.,  2022. [Online].  Available: https://openreview.net/forum?id= L-ceBdl2DPb
- [24] D. Tsur, Z. Goldfeld, and K. Greenewald, 'Max-Sliced Mutual Information,' in Thirty-seventh Conference on Neural Information Processing Systems ,  2023. [Online].  Available: https://openreview.net/forum?id= ce9B2x3zQa
- [25] A. Fayad and M. Ibrahim, 'On Slicing Optimality for Mutual Information,' in Thirty-seventh Conference on Neural Information Processing Systems ,  2023. [Online].  Available: https://openreview.net/forum?id= JMuKfZx2xU
- [26] S.  Wongso,  R.  Ghosh,  and  M.  Motani,  'Understanding  Deep  Neural  Networks  Using  Sliced  Mutual Information,' in 2022 IEEE International Symposium on Information Theory (ISIT) ,  2022, pp. 133-138. doi: 10.1109/ISIT50566.2022.9834357.
- [27] S. Wongso, R. Ghosh, and M. Motani, 'Using Sliced Mutual Information to Study Memorization and Generalization in Deep Neural Networks,' in Proceedings of The 26th International Conference on Artificial Intelligence and Statistics , F. Ruiz, J. Dy, and J.-W. van de Meent, Eds., in Proceedings of Machine Learning Research, vol. 206. PMLR,  2023, pp. 11608-11629. [Online].  Available: https://proceedings. mlr.press/v206/wongso23a.html
- [28] S.  Wongso,  R.  Ghosh,  and  M.  Motani,  'Pointwise  Sliced  Mutual  Information  for  Neural  Network Explainability,' in 2023 IEEE International Symposium on Information Theory (ISIT) ,  2023, pp. 17761781. doi: 10.1109/ISIT54713.2023.10207010.
- [29] J. Dentan, D. Buscaldi, A. Shabou, and S. Vanier, 'Predicting and analyzing memorization within finetuned Large Language Models.' [Online]. Available: https://arxiv.org/abs/2409.18858
- [30] S. Wongso, R. Ghosh, and M. Motani, 'Sliced Information Plane for Analysis of Deep Neural Networks,' Jan. 2025, doi: 10.36227/techrxiv.173833980.08812687/v1.
- [31] K. Nadjahi, K. Greenewald, R. B. Gabrielsson, and J. Solomon, 'Slicing Mutual Information Generalization Bounds for Neural Networks,' in ICML 2023 Workshop Neural Compression: From Information Theory to Applications ,  2023. [Online].  Available: https://openreview.net/forum?id=cbLcwK3SZi
- [32] Z. Hu, S. Kang, Q. Zeng, K. Huang, and Y. Yang, 'InfoNet: Neural Estimation of Mutual Information without Test-Time Optimization,' in Forty-first International Conference on Machine Learning ,  2024. [Online].  Available: https://openreview.net/forum?id=40hCy8n5XH 469 470 471
- [33] T.  Nuradha  and  Z.  Goldfeld,  'Pufferfish  Privacy:  An  Information-Theoretic  Study,' IEEE  Trans.  Inf. Theor. , vol. 69, no. 11, pp. 7336-7356, Nov. 2023, doi: 10.1109/TIT.2023.3296288. 472 473
- [34] I. Butakov et al. , 'MUTINFO.' [Online]. Available: https://github.com/VanessB/mutinfo 474

- [35] A. Kraskov, H. Stögbauer, and P. Grassberger, 'Estimating mutual information,' Phys. Rev. E , vol. 69, no. 6, p. 66138, Jun. 2004, doi: 10.1103/PhysRevE.69.066138. 475 476
- [36] F. Czyż Pawełand Grabowski, J. Vogt, N. Beerenwinkel, and A. Marx, 'Beyond Normal: On the Evaluation of Mutual Information Estimators,' in Advances in Neural Information Processing Systems ,  A.  Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., Curran Associates, Inc.,  2023, pp. 16957-16990. [Online].  Available: https://proceedings.neurips.cc/paper\_files/paper/2023/file/36b80eae 70ff629d667f210e13497edf-Paper-Conference.pdf 477 478 479 480 481
- [37] T. M. Cover and J. A. Thomas, Elements of Information Theory (Wiley Series in Telecommunications and Signal Processing) . USA: Wiley-Interscience, 2006. 482 483

484

485

486

- [38] M. D. Donsker and S. R. Varadhan, 'Asymptotic evaluation of certain markov process expectations for large time. IV,' Communications on Pure and Applied Mathematics , vol. 36, no. 2, pp. 183-212, Mar. 1983, doi: 10.1002/cpa.3160360204.

487

488

489

490

- [39] M. I. Belghazi et al. , 'Mutual Information Neural Estimation,' in Proceedings of the 35th International Conference  on  Machine  Learning ,  J.  Dy  and  A.  Krause,  Eds.,  in  Proceedings  of  Machine  Learning Research, vol. 80. PMLR,  2018, pp. 531-540. [Online].  Available: https://proceedings.mlr.press/v80/ belghazi18a.html
- [40] L. Deng, 'The mnist database of handwritten digit images for machine learning research,' IEEE Signal Processing Magazine , vol. 29, no. 6, pp. 141-142, 2012. 491 492
- [41] A.  van  den  Oord,  Y.  Li,  and  O.  Vinyals,  'Representation  Learning  with  Contrastive  Predictive Coding.' [Online]. Available: https://arxiv.org/abs/1807.03748 493 494
- [42] A. Edelman and B. D. Sutton, 'The beta-Jacobi matrix model, the CS decomposition, and generalized singular value problems,' Foundations of Computational Mathematics , vol. 8, no. 2, pp. 259-285, 2008. 495 496
- [43] A. McBride, 'Special functions, by George E. Andrews, Richard Askey and Ranjan Roy. Pp. 664.£ 60. 1999. ISBN 0 521 62321 9 (Cambridge University Press.),' The Mathematical Gazette , vol. 83, no. 497, pp. 355-357, 1999. 497 498 499
- [44] N. Elezovic, C. Giordano, and J. Pecaric, 'The best bounds in Gautschi's inequality,' Math. Inequal. Appl , vol. 3, no. 2, pp. 239-252, 2000. 500 501
- [45] D. P. Kingma and J. Ba, 'Adam: A Method for Stochastic Optimization.' 2017. 502

## A Supplementary theory 503

Lemma A.1. (Example 2.4 in [1]) 𝗁(𝒩(𝜇,Σ)) = 1 2 log((2𝜋𝑒) 𝑑 det Σ) . 504

Corollary A.2. For (𝑋,𝑌 ) ∼ 𝒩(𝜇, Σ) with non-singular Σ 505

<!-- formula-not-decoded -->

where Σ𝑋 , Σ𝑌 are marginal covariances, Σ𝑋𝑌 is cross-covariance, 𝑑 = min(𝑑 𝑥 , 𝑑 𝑦 ) , and {𝜌 𝑖 } 𝑑 𝑖=1 are singular values of Σ - 1 2 𝑋 Σ𝑋𝑌 Σ - 1 2 𝑌 .

Proof of Corollary A.2. Combining Lemma A.1 and (2) yields the first result. Now note that

<!-- formula-not-decoded -->

where Udiag(𝜌 𝑖 )V 𝖳 is the SVD of Σ - 1 2 𝑋 Σ𝑋𝑌 Σ - 1 2 𝑌 . However,

<!-- formula-not-decoded -->

from which we arrive at the second expression.

□

Lemma A.3. Let A ∈ ℝ 𝑛×𝑚 be full column-rank matrix and Θ ∼ 𝜇 St(𝑛,𝑘) Then Θ 𝖳 A is full-rank with probability one.

Proof of Lemma A.3. Performing QR decomposition of A yields Θ 𝖳 A = Θ 𝖳 QR = d Θ 𝖳 ( I𝑚 0 )R . Since A is full-rank, R is invertible and rank Θ 𝖳 A = rank Θ 𝖳 ( I𝑚 0 ) . Therefore,

<!-- formula-not-decoded -->

□

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

Lemma A.4. (Theorem 1.5 in [42]) Let W∼𝜇O(𝑑) and partition

<!-- formula-not-decoded -->

with W11 of size 𝑘 by 𝑘 . Then the eigenvalues {𝜆 𝑖 } 𝑘 𝑖=1 of W11 W 𝖳 11 follow the Jacobi ensemble

<!-- formula-not-decoded -->

with parameters 𝑎 = 0, 𝑏 = 𝑑 - 2𝑘 , and 𝛽 = 1 (over ℝ ).

Proof of Lemma A.3. Let A1 ∈ ℝ 𝑘×𝑑 and A2 ∈ ℝ (𝑑-𝑘)×𝑑 be independent matrices with i.i.d. entries from 𝒩(0,1) .  By  stacking A1 atop A2 and  then  performing  a  block  QR  decomposition  on  the resulting  Gaussian  matrix,  the  orthogonal  invariance  of  the  Gaussian  law  implies  that  the  two Q-blocks are independent of the upper-triangular factor R , with Q1 and Q2 uniformly distributed on O(𝑘) and St(𝑘, 𝑑 - 𝑘) , respectively. Finally, computing the SVD of the block rows together with R yields the generalized singular value decomposition (GSVD) of the pair (A 1 , A 2 ) : 520 521 522 523 524 525

<!-- formula-not-decoded -->

where U1 ∈ O(𝑘), U 2 ∈ O(𝑑 - 𝑘), ̃ V ∈ O(𝑘) , and C = diag(𝑐 𝑖 ) , S = diag(𝑠 𝑖 ) with 𝑐 𝑖 ≥ 0 , 𝑠 𝑖 ≥ 0 , and 𝑐 2 𝑖 +𝑠 2 𝑖 = 1 for all 𝑖 . The diagonal entries of ̃ C are known as the generalized singular values of the pair (A 1 , A 2 ) . 526 527 528

For a matrix P = diag(𝑝 1 , …, 𝑝 𝑘 ) with i.i.d. 𝑝 𝑖 sampled uniformly from {-1,1} , we have Q1 𝑃 = d W11 . Let W11 = UCV 𝖳 be the SVD of W11 , then one has 529 530 531

<!-- formula-not-decoded -->

Since U1 , ̃ V , and U,V are uniformly distributed and independent of ̃ C, C , we have ̃ C = d C by the invariance of the Haar measure under orthogonal transformations. On the other hand, the generalized singular values ̃ C of a pair (A 1 , A 2 ) follow the law of the Jacobi ensemble with parameters 𝑎 = 0, 𝑏 = 𝑑 - 2𝑘 , and 𝛽 = 1 (Proposition 1.2 in [42]). Therefore, the squared singular values of W11 follow the Jacobi ensemble with the same parameters. □

Corollary A.5. The squared inner product |𝜃 𝖳 𝜙| 2 between two independent random vectors 𝜃, 𝜙 ∼ 𝜇𝕊 𝑑-1 follows Beta( 1 2 , 𝑑-1 2 ) .  Moreover,  the  shifted  inner  product (1 + 𝜃 𝖳 𝜙)/2 is  symmetrically distributed as Beta( 𝑑-1 2 , 𝑑-1 2 ) .

Proof of Corollary A.5. Setting Jacobi parameters 𝑘 = 1, 𝑎 = 0, 𝑏 = 𝑑 - 2 and 𝛽 = 1 , the density is proportional to 𝑥 -1/2 (1 - 𝑥) (𝑑-3)/2 on [0, 1] , which matches the Beta( 1 2 , 𝑑-1 2 ) distribution.

Next, observe that 𝜃 𝖳 𝜙 has a density proportional to (1 - 𝑡) 𝑑-3 2 for 𝑡 ∈ [-1, 1] . Under the change of variables 𝜂 ∼ Beta( 𝑑-1 2 , 𝑑-1 2 ) .

□

532

533

534

535

536

537

538

539

540

541

542

543

544

545

546

547

548

549

550

## B Complete proofs

Proof of Lemma 4.1. One can acquire 𝖨(𝑋; 𝑌 ) = 𝑑 2 log(1 - 𝜌 2 ) from a general expression for MI of two jointly Gaussian random vectors (see Corollary A.2).

Recall that (𝜃 𝖳 𝑋,𝜙 𝖳 𝑌 ) is also Gaussian with cross-covariance 𝜌 𝜃 𝖳 𝜙 . Therefore, by Corollary A.2 we have

<!-- formula-not-decoded -->

From Corollary A.5, we note that |𝜃 𝖳 𝜙| 2 ∼ Beta( 1 2 , 𝑑-1 2 ) , so 551

<!-- formula-not-decoded -->

where the last equality follows from the identity log(1 - 𝑧) = -𝑧 2𝐹 1 (1, 1; 2; 𝑧) with hypergeometric function 2𝐹 1 . Appling Euler's integral transform ([43], Eq. (2.2.3)) gives 552 553

<!-- formula-not-decoded -->

Here 3𝐹 2 denotes the generalized hypergeometric function. 554

Finally, we calculate the limit of 𝖲𝖨(𝑋; 𝑌 ) as 𝜌 2 →1 using properties of beta-distribution. Denoting 𝜂 = (1 + 𝜃 𝖳 𝜙)/2 ∼ Beta( 𝑑-1 2 , 𝑑-1 2 ) (see Corollary A.5), we get 555 556

557

558

559

560

561

562

563

569

570

571

572

573

574

575

576

577

578

579

□

Proof of Proposition 4.4. Using Lemma A.3 and 𝑑 𝑥 , 𝑑 𝑦 &lt; 𝑘 , we get that Θ 𝖳 A and Φ 𝖳 B are injective with probability one for independent Θ,Φ distributed uniformly on St(𝑑 𝑥 , 𝑘) and St(𝑑 𝑦 , 𝑘) . Therefore, according to Theorem 3.1, [𝖨(Θ 𝖳 A𝑋;Φ 𝖳 B𝑌 ) | Θ, Φ] = 𝖨(𝑋; 𝑌 ) almost sure. As a result, 𝖲𝖨 𝑘 (A𝑋;B𝑌 ) = 𝖨(Θ 𝖳 A𝑋; Φ 𝖳 B𝑌 | Θ, Φ) = 𝖨(𝑋; 𝑌 ) . □

Proof of Proposition 4.7. Direct corollary of Corollary A.2.

## C Additional experiments

In this section, we conduct supplementary experiments to evaluate SMI under a broader range of setups. We begin by assessing 𝑘 -SMI on the same set of benchmarks from Section 5. The results for 𝑘 = 1, 2, 3 are presented in Figure 3, Figure 6, and Figure 7, respectively. Notably, saturation remains consistent even for 𝑘 = 𝑑 - 1 (i.e., when only one component is discarded).

Next, we examine a setup involving randomized distribution parameters, following the methodology of [34]. Among other adjustments, this includes randomizing per-component mutual information (e.g.,  assigning  interactions  unevenly  in  this  experiment).  In  some  cases  (e.g.,  the  log-gamma580 581 582

□

<!-- formula-not-decoded -->

where 𝜓 is the digamma function. Using the bounds on digamma function [44]

<!-- formula-not-decoded -->

we derive an upper bound on this expression:

<!-- formula-not-decoded -->

To simplify the bound, one can note that 1 + 𝑒 𝜓(0) &lt; 2 , log(1 + 𝑥) &lt; 𝑥 and 1 𝑑 &lt; 1 𝑑-1 .

Proof of Proposition 4.2.

Let QX,QY ∼ 𝜇 St(𝑘,𝑑) . Then [Q 𝖳 X 𝑋, Q 𝖳 Y 𝑌 ] ∼ 𝒩(0, Σ) , where Σ is a 2𝑘 × 2𝑘 covariance matrix with the following block structure

<!-- formula-not-decoded -->

Using the formula for the determinant of a block matrix Σ yields 564

<!-- formula-not-decoded -->

565

566

567

By the invariance of the Haar measure under left and right multiplication, Q 𝖳 X Q Y = 𝑑 W11 , where W11 is a 𝑘 by 𝑘 left upper block of the matrix W∼𝜇O(𝑑) . According to Lemma A.4, the eigenvalues of W11 W 𝖳 11 follow Jacobi ensemble with parameters 𝑎 = 0, 𝑏 = 𝑑 - 2𝑘 and 𝛽 = 1 :

<!-- formula-not-decoded -->

Thus, we get a general expresion for 𝑘 -SMI 568

<!-- formula-not-decoded -->

□

<!-- image -->

607

608

609

Figure 6: Results of synthetic experiments with different distributions for 2 -SMI. We report mean values and standard deviations computed across 10 runs, with 10 4 samples used for MI estimation and 128 for averaging across projections.

610

611

612

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

exponential distribution), this increases linear redundancy, as component pairs with higher mutual information also exhibit  higher  variance  in  this  particular  scenario.  Our  results  are  displayed  in Figure 8.

Due to numerical constraints, we do not track 𝖨(𝑋; 𝑌 )/𝑑 ,  instead plotting the results against the total mutual information. While this makes saturation slightly less evident, the general trend of SMI decreasing with 𝑑 remains observable. We also highlight the log-gamma-exponential distribution (Figure 8d), where SMI is less prone to saturation under parameter randomization due to the reasons mentioned earlier.

## D Implementation details

## D.1 Synthetic experiments

For the experiments from Section 5 and Section 6.1, we use implementation of Kraskov-StoegbauerGrassberger (KSG) [35] mutual information estimator and random slicing from [34]. The number of neighbors is set to 𝑘 NN = 1 for the KSG estimator. For each configuration, we conduct 10 independent runs with different random seeds to compute means and standard deviations. Our experiments use 10 4 samples for (𝑋,𝑌 ) and 128 samples for (Θ, Φ) .

For the experiments from Section 5, we use independent components with equally distributed percomponent MI. For the supplementary experiments from Figure 8, parameters of each distribution (e.g., covariance matrices) are randomized via the algorithm implemented in [34]. This includes randomization of per-component MI (which is done using a uniform distribution over a (𝑑 - 1) -625 626 627 628

629

dimensional simplex).

<!-- image -->

654

655

656

Figure 7: Results of synthetic experiments with different distributions for 3 -SMI. We report mean values and standard deviations computed across 10 runs, with 10 4 samples used for MI estimation and 128 for averaging across projections.

657

658

659

660

661

662

663

664

665

666

667

For the experiments, we used AMD EPYC 7543 CPU, one core per distribution. Each experiment (fixed 𝑘 , varying 𝑑 ) took no longer then 3 days to compute.

## D.2 Representation learning experiments

For experiments on MNIST dataset, we use a simple ConvNet with three convolutional and two fully connected layers. A three-layer fully-connected perceptron serves as a critic network for the InfoNCE loss. We provide the details in Table 2. We use additive Gaussian noise with 𝜎 = 0.2 as an input augmentation. Training hyperparameters are as follows: batch size = 512, 2000 epochs, Adam optimizer [45] with learning rate 10 -3 .

For the experiments, we used Nvidia A100 GPUs. Each experiment took no longer then 1 day to compute.

Table 2: The NN architectures used to conduct the tests on MNIST images in Section 6.2.

| 668             | NN                          | Architecture                                                                                                                                                                                                                                                             |
|-----------------|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 669 670 671 672 | ConvNet, 24 ×24 images      | ×1 : Conv2d(1, 32, ks=3), MaxPool2d(2), BatchNorm2d, LeakyReLU(0.01) ×1 : Conv2d(32, 64, ks=3), MaxPool2d(2), BatchNorm2d, LeakyReLU(0.01) ×1 : Conv2d(64, 128, ks=3), MaxPool2d(2), BatchNorm2d, LeakyReLU(0.01) ×1 : Dense(128, 128), LeakyReLU(0.01), Dense(128, dim) |
| 673 674         | Critic NN, pairs of vectors | ×1 : Dense(dim + dim, 256), LeakyReLU(0.01) ×1 : Dense(256, 256), LeakyReLU(0.01), Dense(256, 1)                                                                                                                                                                         |

<!-- image -->

700

701

702

Figure 8: Results of synthetic experiments with different distributions. We report mean values and standard deviations computed across 10 runs, with 10 4 samples used for MI estimation and 128 for averaging across projections.

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

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [YES]

Justification:  We  state  our  claims  clearly  in  the  abstract  and  introduction.  The  claims  are supported by theoretical analysis and various experiments.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [YES]

Justification: We discuss limitations in Section 7.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate 'Limitations' section in their paper.
- The  paper  should  point  out  any  strong  assumptions  and  how  robust  the  results  are  to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

750

751

752

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

- Answer: [YES] 753
- Justification: We provide comprehensive statements for theorems and lemmas. We also provide complete proofs in Section B. 754 755
- Guidelines: 756

757

- The answer NA means that the paper does not include theoretical results.

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

- All  the  theorems,  formulas,  and  proofs  in  the  paper  should  be  numbered  and  crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? 768 769 770

Answer: [YES] 771

- Justification:  We  provide  complete  setup  descriptions  for  the  experiments  in  corresponding sections. 772 773

Guidelines: 774

775

- The answer NA means that the paper does not include experiments.

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

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice,  or  if  the  contribution  is  a  specific  model  and  empirical  evaluation,  it  may  be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

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

826

827

828

829

830

831

832

833

834

- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [YES]

Justification: We use openly accessible data only for our experiments. Community-provided code has been used for our experiments, and we reference its origin. Finally, we include additional source code in the submission.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions  should  contain  the  exact  command  and  environment  needed  to  run  to reproduce the results. See the NeurIPS code and data submission guidelines ( https:// nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

- Answer: [YES] 835
- Justification: We provide all the necessary details in the corresponding sections, or in Appendix. 836
- Guidelines: 837

838

- The answer NA means that the paper does not include experiments.

839

840

841

842

843

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full  details  can  be  provided  either  with  the  code,  in  appendix,  or  as  supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? 844 845

- 846

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

Answer: [YES]

Justification: We report mean and standard deviation.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer 'Yes' if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability  that  the  error  bars  are  capturing  should  be  clearly  stated  (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources  (type  of  compute  workers,  memory,  time  of  execution)  needed  to  reproduce  the experiments? 869 870 871

- Answer: [YES] 872
- Justification: We describe our setup and computational load of the experiments. 873
- Guidelines: 874

875

- The answer NA means that the paper does not include experiments.

876

877

878

879

880

881

882

883

884

885

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines

- 886

Answer: [YES]

- Justification: 887

Guidelines: 888

889

890

891

892

893

894

895

896

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If  the  authors  answer  No,  they  should  explain  the  special  circumstances  that  require  a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] 897

898

899

Justification: Although the paper may question methodology of some other works, there are no broader impacts of the research conducted.

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

928

929

930

931

932

933

934

935

936

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples  of  negative  societal  impacts  include  potential  malicious  or  unintended  uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular  applications,  let  alone  deployments.  However,  if  there  is  a  direct  path  to  any negative applications, the authors should point it out. For example, it is legitimate to point out  that  an  improvement  in  the  quality  of  generative  models  could  be  used  to  generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released  models  that  have  a  high  risk  for  misuse  or  dual-use  should  be  released  with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets  that  have  been  scraped  from  the  Internet  could  pose  safety  risks.  The  authors should describe how they avoided releasing unsafe images.

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

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [YES]

Justification: For all the assets, we use citations that the original authors provided.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, https://paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? 961 962

Answer: [NA] 963

Justification: 964

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

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers  should  communicate  the  details  of  the  dataset/code/model  as  part  of  their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

978

Answer: [NA]

Justification: 979

Guidelines: 980

981

982

983

984

985

986

987

988

989

990

991

992

993

994

995

996

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

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional  Review  Board  (IRB)  Approvals  or  Equivalent  for  Research  with  Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial  submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
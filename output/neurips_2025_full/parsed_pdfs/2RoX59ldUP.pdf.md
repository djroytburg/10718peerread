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

## Enhancing Vector Quantization with Distributional Matching: A Theoretical and Empirical Study

## Anonymous Author(s)

Affiliation Address email

## Abstract

The success of autoregressive models largely depends on the effectiveness of vector quantization, a technique that discretizes continuous features by mapping them to the nearest code vectors within a learnable codebook. Two critical issues in existing vector quantization methods are training instability and codebook collapse. Training instability arises from the gradient discrepancy introduced by the straight-through estimator, especially in the presence of significant quantization errors, while codebook collapse occurs when only a small subset of code vectors are utilized during training. A closer examination of these issues reveals that they are primarily driven by a mismatch between the distributions of the features and code vectors, leading to unrepresentative code vectors and significant data information loss during compression. To address this, we employ the Wasserstein distance to align these two distributions, achieving near 100% codebook utilization and significantly reducing the quantization error. Both empirical and theoretical analyses validate the effectiveness of the proposed approach.

## 1 Introduction

Autoregressive models have re-emerged as a powerful paradigm in visual generation, demonstrating significant advances in image synthesis quality. Recent studies [29, 9, 6, 19, 35, 20] highlight that autoregressive approaches now achieve superior results compared to diffusion-based methods [12, 30, 33, 35, 24]. The success of autoregressive visual generative models hinges on the effectiveness of vector quantization (VQ) [36], a technique that compresses and discretizes continuous features by mapping them to the nearest code vectors within a learnable codebook. However, VQ continues to face two major challenges: training instability and codebook collapse.

The first issue originates from the non23 differentiability of VQ, which prevents direct 24 gradient backpropagation from quantized features 25 to their continuous counterparts, thereby hindering 26 effective model optimization. To address this 27 challenge, VQ-VAE [36] introduces a straight28 through estimator (STE) [2]. The STE facilitates 29 gradient propagation by copying the gradients from 30 the quantized features to the continuous features. 31 Nevertheless, the effectiveness of this approach 32 is critically contingent upon the magnitude of the 33 quantization error between the continuous and 34

Distributional Mismatch

<!-- image -->

Distributional Match

Figure 1: The symbols · and × represent the feature and code vectors, respectively. The left figure illustrates the distributional mismatch between the feature and code vectors, while the right figure visualizes their distributional match.

- quantized feature vectors. When the quantization error is excessively large, the training process 35
- becomes notably unstable [19]. 36

- The latter issue emerges due to the inability of existing VQ methods to ensure that all Voronoi cells 1 37 are assigned feature vectors. When only a minority of Voronoi cells are allocated feature vectors, 38 leaving the majority unutilized and unoptimized, severe codebook collapse ensues [42]. Despite 39 considerable research efforts dedicated to mitigating this problem, these methods still exhibit relatively 40 low utilization of code vectors, particularly in scenarios with large codebook sizes [8, 34, 39, 19, 42]. 41 This is due to the fact that, as the codebook size increases, the number of V oronoi cells also increases, 42
- significantly raising the challenge of ensuring that every cell is assigned a feature vector. 43

In this paper, we examine these issues by investigating the distributions of the features and code 44 vectors. To illustrate the idea, Figure 1 presents two extreme scenarios: the left panel depicts a 45 significant mismatch between the two distributions, while the right panel shows a match. In the left 46 panel, all features are mapped to a single codeword, resulting in large quantization errors and minimal 47 codebook utilization. In contrast, the right panel demonstrates that a distributional match leads to 48 negligible quantization error and near 100% codebook utilization. This suggests aligning these two 49 distributions in VQ could potentially address the issues of training instability and codebook collapse. 50

51

52

53

54

55

56

57

58

To investigate the idea above, we first introduce three principled criteria that a VQ method should satisfy. Guided by this criterion triple, we conduct qualitative and quantitative analyses, demonstrating that aligning the distributions of the feature and code vectors results in near 100% codebook utilization and minimal quantization error. Additionally, our theoretical analysis underscores the importance of distribution matching for vector quantization. To achieve this alignment, we employ the quadratic Wasserstein distance which has a closed-form representation under a Gaussian hypothesis. Our approach effectively mitigates both training instability and codebook collapse, thereby enhancing image reconstruction performance in visual generative tasks.

59

## 2 Understanding Distribution Matching

This section introduces a novel distributional perspective for VQ. By defining three principled criteria 60 for VQ evaluation, we empirically and theoretically demonstrate that distribution matching yields an 61 almost optimal VQ solution. 62

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

## 2.1 An Overview of Vector Quantization

As the core component in visual tokenizer [36, 19, 35], VQ acts as a compressor that discretizes continuous latent features into discrete visual tokens by mapping them to the nearest code vectors within a learnable codebook.

Figure 2 illustrates the classic VQ process [36], which consists of an encoder E ( · ) , a decoder D ( · ) , and an updatable codebook { e k } K k =1 ∈ R d containing a finite set of code vectors. Here, K represents the size of the codebook, and d denotes the dimension of the code vectors. Given an image x ∈ R H × W × 3 , the goal is to derive a spatial collection of codeword IDs r ∈ N h × w as image tokens. This is achieved by passing the image through the encoder to obtain

<!-- image -->

z e = E ( x ) ∈ R h × w × d , followed by a spatial-wise quantizer Q ( · ) that maps each spatial feature z ij e 77 to its nearest code vector e k : 78

<!-- formula-not-decoded -->

These tokens are then used to retrieve the corresponding codebook entries z ij q = Q ( z ij e ) = e r ij , 79 which are subsequently passed through the decoder to reconstruct the image as ̂ x = D ( z q ) . Despite 80 its success in high-fidelity image synthesis [36, 29, 9], VQ faces two key challenges: training 81 instability and codebook collapse. 82

1 A comprehensive understanding of codebook collapse through the lens of Voronoi partition is provided in Appendix C.

Training Instability This issue occurs because during backpropagation, the gradient of z q cannot 83 flow directly to z e due to the non-differentiable function Q . To optimize the encoder's network param84 eters through backpropagation, VQ-VAE [36] employs the straight-through estimator (STE) [3], which 85 copies gradients directly from z q to z e . However, this approach carries significant risks-especially 86 when z q and z e are far apart. In these cases, the gradient gap between the representations can 87 grow substantially, destabilizing the training process. In this paper, we tackle the training instability 88 challenge from a distributional viewpoint. 89

90

91

92

93

94

95

Codebook Collapse Codebook collapse occurs when only a small subset of code vectors receives optimization-useful gradients, while most remain unrepresentative and unupdated [8, 34, 39, 19, 42]. Researchers have proposed various solutions to this problem, such as improved codebook initialization [43], reinitialization strategies [8, 38], and classical clustering algorithms like k -means [5] and k -means++[1] for codebook optimization [29, 42]. Beyond these deterministic approaches that select the best-matching token, researchers have also explored stochastic quantization strategies [40, 28, 34].

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

However, these methods still exhibit relatively low utilization of code vectors, particularly with large codebook sizes K [42, 25]. In this paper, we address this issue by the distributional matching between feature vectors and code vectors.

## 2.2 Evaluation Criteria

Given a set of feature vectors { z i } N i =1 sampled from feature distribution P A and code vectors { e k } K k =1 sampled from codebook distribution P B , vector quantization involves finding the nearest, and thus most representative, code vector for each feature vector:

<!-- formula-not-decoded -->

The original feature vector z i is then quantized to z ′ i . Below, we introduce three key criteria to evaluate this process.

Criterion 1 (Quantization Error) . The quantization error measures the average distortion introduced by VQ and is defined as

<!-- formula-not-decoded -->

A smaller E signifies a more accurate quantization of the original feature vectors, resulting in a smaller gradient gap between z i and z ′ i . Consequently, a small E suggests that the issue of training instability can be effectively mitigated.

Criterion 2 (Codebook Utilization Rate) . The codebook utilization rate measures the proportion of code vectors used in VQ and is defined as

<!-- formula-not-decoded -->

A higher value of U reduces the risk of codebook collapse. Ideally, U should reach 100 % , indicating that all code vectors are utilized. As discussed in Appendix D, U can only measure the completeness of codebook utilization; it does not suffice to evaluate the degree of codebook collapse. This motivates us to introduce the codebook perplexity criterion.

Criterion 3 (Codebook Perplexity) . The codebook perplexity measures the uniformity of codebook utilization in VQ and is defined as

<!-- formula-not-decoded -->

where p k = 1 N ∑ N i =1 1 ( z ′ i = e k ) . Ahigher value of C indicates that code vectors are more uniformly 118 selected in the VQ process. Ideally, C reaches its maximum at C 0 = exp( -∑ K k =1 1 K log 1 K ) = K 119 when code vectors are completely uniformly utilized. Therefore, as a complementary measure to 120 Criterion 2, the combination of U and C can effectively evaluate the degree of codebook collapse. 121

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

151

Figure 3: Qualitative analyses of the criterion triple ( E , U , C ) : The red and green disks represent the uniform distributions of feature vectors and code vectors, respectively.

<!-- image -->

We refer to ( E , U , C ) as the criterion triple. When comparing extreme cases of distributional match and mismatch shown in Figure 1, we find that distributional matching significantly outperforms mismatching across all three criteria. Using this criterion triple, we present detailed analyses that demonstrate the advantages of distribution matching.

## 2.3 The Effects of Distribution Matching

We conduct a simple synthetic experiment to provide intuitive insights (See experimental details in Appendix I.1). Specifically, we assume that the distributions P A and P B are uniform distributions confined within two distinct disks, as depicted in Figure 3. We then sample a set of feature vectors { z i } N i =1 uniformly from the red disk, and a set of code vectors { e k } K k =1 uniformly from the green circle. The criterion triple ( E , U , C ) is then calculated based on the definitions in Criteria 1 to 3.

We examine two cases. The first involves two disks with identical radii but different centers. As shown in Figures 3(a) to 3(d), when the centers of the disks move closer together, the criterion triple improves toward optimal values. Specifically, E decreases from 1.19 to 0.05, U rises from 2% to 100% , and C increases from 3.8 to 344.9.

The second case shows two distributions with identical centers but different radii. When the codebook distribution's support lies within the feature distribution's support (as shown in Figures 3(e) and 3(f)), it results in a notably larger E , slightly lower U , and significantly smaller C compared to the aligned distributions shown in Figure 3(d). Conversely, when the codebook distribution's support extends beyond the feature distribution's support, E shows a modest increase while both U and C decrease significantly, as illustrated in Figures 3(g) and 3(h). We provide detailed explanations of these experimental results in Appendix E.

From both cases, we can conclude that the VQ achieves the optimal criterion triple when the feature and codebook distributions are identical. This observation will be further supported by more quantitative analyses in Appendix F.

## 2.4 Theoretical Analyses

In this section, we provide theoretical evidence to support our empirical observations. Let the code vectors { e k } K k =1 and feature vectors { z i } N i =1 be independently and identically drawn from P B and P A , respectively. We say a codebook { e k } K k =1 attains full utilization asymptotically with respect to { z i } N i =1 if the codebook utilization rate U ( { e k } K k =1 ; { z i } N i =1 ) tends to 1 in probability as N approaches infinity:

<!-- formula-not-decoded -->

For the codebook distribution P B , we say it attains full utilization asymptotically with respect 152 to P A if, with probability 1, the randomly generated codebook { e k } K k =1 achieves full utilization 153 asymptotically. 154

Additionally, a codebook distribution P B is said to have vanishing quantization error asymptotically 155 with respect to a domain Ω ⊆ R d if the quantization error over all data of size N tends to zero in 156 probability as K approaches infinity: 157

<!-- formula-not-decoded -->

Our first theorem shows that supp( P A ) = supp( P B ) is sufficient and necessary for the codebook distribution P B to attain both full utilization and vanishing quantization error asymptotically. For simplicity, P A is assumed to have a density function f A with bounded support Ω ⊆ R d .

Theorem 1. Assume Ω = supp( P A ) is a bounded open area. The codebook distribution P B attains full utilization and vanishing quantization error asymptotically if and only if supp( P B ) = supp( P A ) , where S denotes the closure of the set S .

Theorem 1 establishes the optimal support of the codebook distribution. The boundedness of Ω is required as we consider the worst case quantization error in equation 2. In real applications, when P A follows an absolutely continuous distribution over an unbounded domain, then { z i } N i =1 generated from P A will be bounded with high probability. Thus, Theorem 1 also provides theoretical insights for a target distribution P A with an unbounded domain.

Besides the optimal support, we also determine the optimal density of the codebook distribution by invoking existing results characterizing asymptotic optimal quantizers [10]. Specifically, we consider the case where N approaches to infinity and define the expected quantization error of a codebook { e k } with respect to P A as

<!-- formula-not-decoded -->

Acodebook { e ∗ k } K k =1 is called the set of optimal centers for P A if it achieves the minimal quantization error:

<!-- formula-not-decoded -->

Theorem 2 demonstrates that, under weak regularity conditions, the empirical measure of the optimal centers for P A converges in distribution to a fixed distribution determined by P A . Notably, we do not assume a bounded domain in the following theorem.

Theorem 2 (Theorem 7.5, [10]) . Suppose Z ∼ P A is absolutely continuous with respect to the Lesbegue measure in R d and E ∥ Z ∥ 2+ δ &lt; ∞ for some δ &gt; 0 . Then the empirical measure of the optimal centers for P A ,

<!-- formula-not-decoded -->

converges weakly to a fixed distribution P ∗ A , whose density function f ∗ A is proportional to f ( d +2) /d A .

Theorem 2 implies that P B = P ∗ A is the optimal codebook distribution in the asymptotic regime as K approaches infinity. In high-dimensional spaces with large d , this optimal distribution P B = P ∗ A closely approximates P A . This further motivates us to align the codebook distribution P B with the feature distribution P A .

## 3 Methodology

In this section, we introduce the quadratic Wasserstein distance for distributional matching between features and the codebook. We then apply this technique to two frameworks.

## 3.1 Distribution Matching via Wasserstein Distance

We assume a Gaussian hypothesis for the distributions of both the feature and code vectors. For computational efficiency, we employ the quadratic Wasserstein distance, as defined in Appendix B, to align these two distributions. Although other statistical distances, such as the Kullback-Leibler divergence [17, 12], are viable alternatives, they lack simple closed-form representations, making them computationally expensive. The following lemma provides the closed-form representation for the quadratic Wasserstein distance between two Gaussian distributions.

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

Figure 4: Illustration of the Wasserstein VQ . The architecture integrates an encoder-decoder network with a VQ module. In the VQ module, we augment the vanilla VQ framework [36] by incorporating our proposed Wasserstein loss L W to achieve distributional matching between features z e ( z ij e ∼ P A ) and the codebook e k ( e k ∼ P B ). This enhancement leads to 100% codebook utilization and the minimal achievable quantization error between z e and z q .

<!-- image -->

Lemma 3 ([27]) . The quadratic Wasserstein distance between N ( µ 1 , Σ 1 ) and N ( µ 2 , Σ 2 ) 196

<!-- formula-not-decoded -->

The lemma above indicates that the quadratic Wasserstein distance can be easily computed using the population means and covariance matrices. In practice, we estimate these population quantities, µ 1 , µ 2 , Σ 1 , and Σ 2 , with their sample counterparts: ̂ µ 1 , ̂ µ 2 , ̂ Σ 1 , and ̂ Σ 2 . The empirical quadratic Wasserstein distance is then used as the optimization objective to align the feature and codebook distributions:

<!-- formula-not-decoded -->

A smaller value of L W indicates stronger alignment between the feature distribution P A and the codebook distribution P B . We refer to the VQ algorithm that employs L W as Wasserstein VQ .

## 3.2 Integration into the VQ-VAE Framework

We first examine Wasserstein VQ within the VQ-VAE framework [36]. As illustrated in the Figure 4, the VQ-VAE model combines three key components: an encoder E ( · ) , a decoder D ( · ) , a quantizer Q ( · ) with a learnable codebook { e k } K k =1 . As described earlier in Section 2.1, for an input image x , the encoder processes the image to yield a spatial feature z e = E ( x ) ∈ R h × w × d . The quantizer converts z e into a quantized feature z q , from which the decoder reconstructs the image as ̂ x = D ( z q ) . By incorporating our proposed Wasserstein loss L W into the VQ-VAE framework, the overall loss objective can be formulated as follows:

<!-- formula-not-decoded -->

where sg denotes the stop-gradient operation. β and γ are hyper-parameters. We set γ = 0 . 5 for all experiments.

## 3.3 Integration into the VQGAN Framework

To ensure high perceptual quality in the reconstructed images, we further investigate Wasserstein VQ 215 within the VQGAN framework [9]. VQGAN extends the VQ-VAE framework by integrating a VGG 216 network [32] and a patch-based discriminator [9, 15]. The overall training objective of VQGAN can 217 be written as follows: 218

<!-- formula-not-decoded -->

Where L Per and L GAN denote the VGG-based perceptual loss [41], and GAN loss [14, 21], respectively. 219 We set λ = 0 . 2 for all experiments. 220

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

Table 1: Comparison of VQ-VAEs trained on FFHQ dataset following [36].

| Approach       |   Tokens |   Codebook Size | U ( ↑ )   |   C ( ↑ ) |   PSNR( ↑ ) |   SSIM( ↑ ) |   Rec. Loss ( ↓ ) |
|----------------|----------|-----------------|-----------|-----------|-------------|-------------|-------------------|
| Vanilla VQ     |      256 |           16384 | 3.8%      |     527.2 |       27.83 |        73.8 |            0.0119 |
| EMA VQ         |      256 |           16384 | 14.0%     |    1795.7 |       28.39 |        74.8 |            0.0106 |
| Online VQ      |      256 |           16384 | 11.7%     |    1115.3 |       27.68 |        72.6 |            0.0125 |
| Wasserstein VQ |      256 |           16384 | 100%      |   15713.3 |       29.03 |        76.6 |            0.0093 |
| Vanilla VQ     |      256 |           50000 | 1.2%      |     516.8 |       27.83 |        73.6 |            0.012  |
| EMA VQ         |      256 |           50000 | 10.3%     |    4075.7 |       28.61 |        75.3 |            0.0101 |
| Online VQ      |      256 |           50000 | 6.0%      |    1642.9 |       28.37 |        74.6 |            0.0107 |
| Wasserstein VQ |      256 |           50000 | 100%      |   47496.4 |       29.24 |        77   |            0.0089 |
| Vanilla VQ     |      256 |          100000 | 0.6%      |     481   |       27.86 |        74.2 |            0.0118 |
| EMA VQ         |      256 |          100000 | 2.7%      |    2087.5 |       28.43 |        74.8 |            0.0105 |
| Online VQ      |      256 |          100000 | 3.6%      |    1556.8 |       27.12 |        71.1 |            0.0142 |
| Wasserstein VQ |      256 |          100000 | 100%      |   93152.7 |       29.53 |        78   |            0.0083 |

Table 2: Comparison of VQ-VAEs trained on ImageNet dataset following [36].

| Approach       |   Tokens |   Codebook Size | U ( ↑ )   |   C ( ↑ ) |   PSNR( ↑ ) |   SSIM( ↑ ) |   Rec. Loss ( ↓ ) |
|----------------|----------|-----------------|-----------|-----------|-------------|-------------|-------------------|
| Vanilla VQ     |      256 |           16384 | 2.5%      |     360.7 |       24.44 |        57.5 |            0.0294 |
| EMA VQ         |      256 |           16384 | 14.5%     |    1861.5 |       24.98 |        59.2 |            0.0267 |
| Online VQ      |      256 |           16384 | 22.2%     |    1465.6 |       24.88 |        58.6 |            0.0273 |
| Wasserstein VQ |      256 |           16384 | 100%      |   15539.1 |       25.47 |        61.2 |            0.0242 |
| Vanilla VQ     |      256 |           50000 | 0.9%      |     378.7 |       24.4  |        57.7 |            0.0295 |
| EMA VQ         |      256 |           50000 | 16.8%     |    6139.3 |       25.37 |        60.9 |            0.0246 |
| Online VQ      |      256 |           50000 | 9.9%      |    2241.7 |       25.09 |        59.7 |            0.026  |
| Wasserstein VQ |      256 |           50000 | 100%      |   46133.2 |       25.72 |        62.3 |            0.023  |
| Vanilla VQ     |      256 |          100000 | 0.4%      |     337   |       24.43 |        57.4 |            0.0295 |
| EMA VQ         |      256 |          100000 | 3.0%      |    2170   |       25.13 |        60.1 |            0.0257 |
| Online VQ      |      256 |          100000 | 4.1%      |    1709.9 |       24.95 |        59.1 |            0.0267 |
| Wasserstein VQ |      256 |          100000 | 100%      |   93264.7 |       25.88 |        63   |            0.0223 |

## 4 Experiments

In this section, we empirically demonstrate the effectiveness of our proposed Wasserstein VQ algorithm in visual tokenization tasks. Our experiments are conducted within the frameworks of VQ-VAE [36] and VQGAN [9]. The PyTorch code, including training environment, scripts and logs, will be made publicly available.

## 4.1 Evaluation on VQ-VAE Framework

Datasets and Baselines Experiments are conducted on four benchmark datasets: two low-resolution datasets, i.e., CIFAR-10 [18] and SVHN [26], and two high-resolution datasets FFHQ [16] and ImageNet [7]. We evaluated our approach against several representative VQ methods: Vanilla VQ [36], EMA VQ [29], which uses exponential moving average updates and is also referred to as k -means, Online VQ, which employs k -means++ in CVQ-VAE [42]. For detailed experimental settings, please refer to Appendix J.

Metrics We employ multiple evaluation metrics, including the Codebook Utilization Rate ( U ), Codebook Perplexity ( C ), peak signal-to-noise ratio (PSNR), patch-level structural similarity index (SSIM), and pixel-level reconstruction loss (Rec. Loss). We exclude the quantization error ( E ) from our reported results, as it is highly sensitive to distribution variances-a factor analyzed in Appendix G. Since these distribution variances remain uncontrolled in our experiments, fair comparison based on ( E ) would be unreliable. To ensure an equitable assessment, Appendix H provides an atomic setting where distribution variances are fully controlled and identical across all VQ variants.

Main Results As shown in Tables 1, 2, and Tables 6, 7 in the Appendix K, our proposed Wasserstein VQ outperforms all baselines on both datasets, achieving superior performance across almost all evaluation metrics under various experimental settings. The underlying reason is that VQ inherently functions as a compressor, transitioning from a continuous latent space to a discrete space, where minimal information loss indicates improved expressivity. Our proposed Wasserstein VQ employs explicit distribution matching constraints, thereby achieving a more favorable alignment between

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

283

284

285

286

Figure 5: Visualization of feature and codebook distributions. The symbols blue · and red × represent the feature and code vectors, respectively.

<!-- image -->

the feature vectors and code vectors. This results in nearly 100% codebook utilization and almost minimal quantization error, leading to the lowest Rec. Loss among all settings.

Representation Visualization To visualize the distributions of feature vectors and code vectors across different VQ methods trained on the FFHQ dataset (with a fixed codebook size of 8192 ), we randomly sample 3000 feature vectors and 1000 code vectors and plot their scatter diagrams. As shown in Figure 5(a) and Figure 5(b), in Vanilla VQ and EMA VQ, the majority of code vectors are clustered near the zero point, rendering them effectively unusable. While Online VQ avoids this central clustering issue, most of its code vectors are distributed at the two extremes of the feature space, as illustrated in Figure 5(c). This distributional mismatch leads to increased information loss and reduced codebook utilization. In contrast to these three VQ methods, Wasserstein VQ demonstrates significantly better distributional matching between feature vectors and code vectors. This alignment substantially minimizes information loss and enhances codebook utilization.

Gaussian Hypothesis Justification To justify the reasonableness of the Gaussian assumption, we extract feature vectors from the encoder and computed the density of arbitrary two dimensions by binning the data points into 29 groups, as visualized in Figure 6(a). Furthermore, we randomly selected 2000 data points from any two dimensions and plotted them in a scatter plot, as shown in Figure 6(b). It is evident that the feature vectors exhibit Gaussian-like characteristics. The under-

Figure 6: Visualization of feature vectors.

<!-- image -->

lying reason for this behavior can be attributed to the central limit theorem, which posits that learned feature vectors and code vectors will approximate a Gaussian distribution given a sufficiently large sample size and a relatively low-dimensional space, i.e., d = 8 .

Analyses of Codebook Size We investigate the impact of the codebook size K on VQ performance, as presented in Table 1, and Table 8 in Appendix L. Vanilla VQ suffers from severe codebook collapse even with a small K , such as K = 1024 . In contrast, improved algorithms, such as EMA VQ and Online VQ, also experience codebook collapse when K is very large, e.g., K ≥ 50000 . Notably, Wasserstein VQ consistently maintains 100% codebook utilization, regardless of the codebook size. This demonstrates that distributional matching by quadratic Wasserstein distance effectively resolves the issue of codebook collapse.

Analyses of Codebook Dimensionality We further investigate the impact of codebook dimensionality d on VQ performance. We conduct experiments on CIFAR-10 dataset and range d from 2 to 32. As shown in Table 9 in Appendix L our proposed Wasserstein VQ consistently outperforms all baselines regardless of dimensionality. Notably, we observe the curse of dimensionality phenomenon-performance degrades as dimensionality increases. Vanilla VQ exhibits the most severe degradation, followed by EMA VQ and Online VQ, while our Wasserstein VQ shows only minimal codebook utilization reduction.

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

Table 3: Comparison of VQGAN trained on FFHQ dataset following [9].

| Method                       |   Tokens | Codebook Size   | Utilization ( % ) ↑   |   rFID ↓ | LPIPS ↓   | PSNR ↑   | SSIM ↑   |
|------------------------------|----------|-----------------|-----------------------|----------|-----------|----------|----------|
| RQVAE † [19]                 |      256 | 2,048           | -                     |     7.04 | 0.13      | 22.9     | 67.0     |
| VQ-WAE † [37]                |      256 | 1,024           | -                     |     4.2  | 0.12      | 22.5     | 66.5     |
| MQVAE † [13]                 |      256 | 1,024           | 78.2                  |     4.55 | -         | -        | -        |
| VQGAN † [9]                  |      256 | 16,384          | 2.3                   |     5.25 | 0.12      | 24.4     | 63.3     |
| VQGAN-FC † [39]              |      256 | 16,384          | 10.9                  |     4.86 | 0.11      | 24.8     | 64.6     |
| VQGAN-EMA † [29]             |      256 | 16,384          | 68.2                  |     4.79 | 0.10      | 25.4     | 66.1     |
| VQGAN-LC † [43]              |      256 | 100,000         | 99.5                  |     3.81 | 0.08      | 26.1     | 69.4     |
| Wasserstein VQ               |      256 | 16,384          | 100                   |     3.08 | 0.08      | 26.3     | 70.4     |
| ⋆                            |      256 | 50,000          | 100                   |     2.96 | 0.08      | 26.5     | 71.4     |
|                              |      256 | 100,000         | 100                   |     2.71 | 0.07      | 26.6     | 71.9     |
| Multi-scale Wasserstein VQ ⋆ |      680 | 16,384          | 100                   |     2.48 | 0.06      | 27.4     | 74.0     |
|                              |      680 | 50,000          | 100                   |     2.07 | 0.06      | 27.6     | 74.6     |
|                              |      680 | 100,000         | 100                   |     1.79 | 0.05      | 27.9     | 75.4     |

## 4.2 Evaluation on VQGAN Framework

Dataset, Baselines, and Metrics We evaluated our approach against following methods on the FFHQ dataset: RQVAE [19], VQGAN [9], VQGAN-FC [39], VQGAN-EMA [29], VQ-WAE [37], MQVAE [13], and VQGAN-LC [43]. Following VQGAN-LC [43], we employ the Fréchet Inception Distance (r-FID) [11], Learned Perceptual Image Patch Similarity (LPIPS) [41], PSNR, and SSIM to evaluate visual reconstruction quality.

Main Results As presented in Table 3, our proposed Wasserstein VQ outperforms all baselines across all evaluation metrics within the VQGAN framework. This superior performance stems from its VQ system that minimizes information loss, as discussed in Section 4.1, thereby achieving optimal reconstruction fidelity and visual perceptual quality. Notably, when integrating V AR's multi-scale VQ [35] with our Wasserstein VQ , we observe a significant improvement in rFID (reduced from 2.71 to 1.79 with codebook size K = 100000 ). Figure 7 demonstrates that Wasserstein VQ 's reconstructed images exhibit only minimal differences from the inputs, confirming its exceptional visual tokenization capability.

Figure 7: Visualization of reconstructed Images. The top row displays the original input images with a resolution of 256 × 256 pixels, while the bottom row shows the reconstructed images from the Wasserstein VQ .

<!-- image -->

## 5 Conclusion

This paper examines vector quantization (VQ) from a distributional perspective, introducing three 302 key evaluation criteria. Empirical results demonstrate that optimal VQ results are achieved when the 303 distributions of continuous feature vectors and code vectors are identical. Our theoretical analysis 304 confirms this finding, emphasizing the crucial role of distributional alignment in effective VQ. 305 Based on these insights, we propose using the quadratic Wasserstein distance to achieve alignment, 306 leveraging its computational efficiency under a Gaussian hypothesis. This approach achieves near307 full codebook utilization while significantly reducing quantization error. Our method successfully 308 addresses both training instability and codebook collapse, leading to improved downstream image 309 reconstruction performance. A limitation of this work, however, is that our proposed distributional 310 matching approach relies on the assumption of Gaussian distribution, which may not strictly hold 311 in all scenarios. In future work, we aim to develop methods that do not depend on this assumption, 312 thereby broadening the applicability and robustness of our VQ framework. 313

314

315

316

317

318

319

320

## References

- [1] David Arthur and Sergei Vassilvitskii. k-means++: the advantages of careful seeding. In ACM-SIAM Symposium on Discrete Algorithms , 2007.
- [2] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. ArXiv , 2013.
- [3] Yoshua Bengio, Nicholas Léonard, and Aaron C. Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. ArXiv , 2013.
- [4] A. Bhattacharyya. On a measure of divergence between two statistical populations defined by 321 their probability distributions. Bulletin of the Calcutta Mathematical Society , 1943. 322
- [5] Paul S. Bradley and Usama M. Fayyad. Refining initial points for k-means clustering. In ICML , 323 1998. 324
- [6] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked 325 generative image transformer. In CVPR , 2022. 326

327

328

- [7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, K. Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR , 2009.

329

330

331

332

- [8] Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, and Ilya Sutskever. Jukebox: A generative model for music. ArXiv , 2020.
- [9] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis. In CVPR , 2021.
- [10] Siegfried Graf and Harald Luschgy. Foundations of quantization for probability distributions . 333 Springer Science &amp; Business Media, 2000. 334
- [11] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 335 Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS , 336 2017. 337
- [12] Jonathan Ho, Ajay Jain, and P. Abbeel. Denoising diffusion probabilistic models. In NeurIPS , 338 2020. 339
- [13] Mengqi Huang, Zhendong Mao, Quang Wang, and Yongdong Zhang. Not all image regions 340 matter: Masked vector quantization for autoregressive image generation. In CVPR , 2023. 341
- [14] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with 342 conditional adversarial networks. In CVPR , 2017. 343
- [15] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer 344 and super-resolution. In ECCV , 2016. 345
- [16] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative 346 adversarial networks. In CVPR , 2018. 347
- [17] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In ICLR , 2014. 348
- [18] Alex Krizhevsky. Learning multiple layers of features from tiny images. ArXiv , 2009. 349
- [19] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive 350 image generation using residual quantization. In CVPR , 2022. 351
- [20] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Jindong Wang, Zhe Lin, and Bhiksha 352 Raj. Xq-gan: An open-source image tokenization framework for autoregressive generation. 353 ArXiv , 2024. 354
- [21] Jae Hyun Lim and J. C. Ye. Geometric gan. ArXiv , 2017. 355
- [22] David Lindley and Solomon Kullback. Information theory and statistics. Journal of the 356 American Statistical Association , 1959. 357

- [23] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR , 2019. 358
- [24] Xiaoxiao Ma, Mohan Zhou, Tao Liang, Yalong Bai, Tiejun Zhao, H. Chen, and Yi Jin. Star: 359 Scale-wise text-to-image generation via auto-regressive representations. ArXiv , 2024. 360
- [25] Fabian Mentzer, David C. Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar 361 quantization: Vq-vae made simple. In ICLR , 2024. 362
- [26] Yuval Netzer, Tao Wang, Adam Coates, A. Bissacco, Bo Wu, and A. Ng. Reading digits in 363 natural images with unsupervised feature learning. ArXiv , 2011. 364
- [27] Ingram Olkin and Friedrich Pukelsheim. The distance between two random vectors with given 365 dispersion matrices. Linear Algebra and its Applications , 1982. 366
- [28] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark 367 Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML , 2021. 368
- [29] Ali Razavi, Aäron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images 369 with vq-vae-2. In NeurIPS , 2019. 370
- [30] Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High371 resolution image synthesis with latent diffusion models. In CVPR , 2022. 372
- [31] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for 373 biomedical image segmentation. In MICCAI , 2015. 374
- [32] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale 375 image recognition. In ICLR , 2015. 376
- [33] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. 377 Autoregressive model beats diffusion: Llama for scalable image generation. ArXiv , 2024. 378
- [34] Yuhta Takida, Takashi Shibuya, Wei-Hsiang Liao, Chieh-Hsin Lai, Junki Ohmura, Toshimitsu 379 Uesaka, Naoki Murata, Shusuke Takahashi, Toshiyuki Kumakura, and Yuki Mitsufuji. Sq-vae: 380 Variational bayes on discrete representation with self-annealed stochastic quantization. In ICML , 381 2022. 382
- [35] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive 383 modeling: Scalable image generation via next-scale prediction. In NeurIPS , 2024. 384
- [36] Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation 385 learning. In NeurIPS , 2017. 386
- [37] Tung-Long Vuong, Trung-Nghia Le, He Zhao, Chuanxia Zheng, Mehrtash Harandi, Jianfei Cai, 387 and Dinh Q. Phung. Vector quantized wasserstein auto-encoder. In ICML , 2023. 388
- [38] Will Williams, Sam Ringer, Tom Ash, John Hughes, David Macleod, and Jamie Dougherty. 389 Hierarchical quantized autoencoders. In NeurIPS , 2020. 390
- [39] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, 391 Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with 392 improved vqgan. In ICLR , 2022. 393
- [40] Jiahui Zhang, Fangneng Zhan, Christian Theobalt, and Shijian Lu. Regularized vector quantiza394 tion for tokenized image synthesis. In CVPR , 2023. 395
- [41] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreason396 able effectiveness of deep features as a perceptual metric. In CVPR , 2018. 397
- [42] Chuanxia Zheng and Andrea Vedaldi. Online clustered codebook. In ICCV , 2023. 398
- [43] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of vqgan to 399 100,000 with a utilization rate of 99%. ArXiv , 2024. 400
- [44] Yongxin Zhu, Bocheng Li, Yifei Xin, and Linli Xu. Addressing representation collapse in 401 vector quantized models with one linear layer. ArXiv , 2024. 402

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly and accurately articulate the paper's key claims and contributions, notably the proposed distributional matching framework via Wasserstein distance to address training instability and codebook collapse in vector quantization (VQ). These claims are rigorously substantiated by theoretical derivations and empirical validation across the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We explicitly discuss the limitation that their distributional matching approach relies on a Gaussian distribution assumption, noting this may not strictly hold in all practical scenarios. They also indicate that future work will aim to generalize beyond this.

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

Justification: The paper clearly states all assumptions underlying the theoretical results, such as bounded support and continuity conditions. Complete proofs of theoretical claims (e.g., Theorem 1 and Theorem 2) are thoroughly provided in the appendix, alongside rigorous mathematical justifications within the main text.

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

Justification: We provide detailed descriptions of experimental setups, hyperparameters, model architectures, datasets, and training procedures necessary for reproducibility, along with comprehensive results and analyses in the appendices. The authors will make the complete PyTorch implementation publicly available, including the training environment configuration, scripts, and logs in the supplementary material.

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

Justification: The code, training scripts, and logs will be made publicly available on an anonymous repository for scrutiny and reproducibility.

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

Justification: The paper specifies all necessary experimental details clearly, including data splits, hyperparameter settings, optimizer types (e.g., AdamW), and training protocols across multiple benchmark datasets (CIFAR-10, SVHN, FFHQ, ImageNet) to enable full understanding and assessment of the reported results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The experimental results include clearly defined metrics such as Codebook Utilization, Codebook Perplexity, PSNR, SSIM, and reconstruction loss, with multiple runs reported for confidence. Error bars (95% confidence intervals) are explicitly computed and provided, especially for synthetic experiments (see Appendix F)

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

Justification: The paper provides detailed specifications of computational resources, including GPU types, memory requirements, and training durations clearly outlined in the experiment descriptions in the Table 5 in the appendix J.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conforms fully to the NeurIPS Code of Ethics, as it involves algorithmic and synthetic experiments without posing ethical concerns such as privacy violations, fairness issues, or environmental harms. No identifiable data or ethically sensitive methodologies are involved.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper primarily introduces foundational methodological improvements for vector quantization in generative modeling without direct societal implications.

## Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper focuses on fundamental algorithmic improvements in VQ and does not involve releasing pretrained language models, generative models prone to misuse, or large-scale scraped datasets, thus posing no high risks of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

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

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All existing datasets (CIFAR-10, SVHN, FFHQ, ImageNet) used for empirical validation are clearly referenced, properly credited, and publicly available with well-known licenses cited explicitly within the experimental settings.

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

Justification: The paper does not introduce or release new datasets, code packages, or pretrained models as new assets; it leverages well-established datasets and publicly accessible frameworks for validation.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper involves no crowdsourcing, human subject research, or any participant interaction.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The research involves no human subjects, thus IRB or equivalent ethical review is not applicable.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The methodology does not involve using large language models (LLMs) as part of the core methodological contribution; any usage would be solely related to standard writing or formatting purposes.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Optimal Support of The Codebook Distribution 757

Proof of Theorem 1. First, we assume supp( P B ) = supp( P A ) . Then for any z ∈ supp( P A ) , 758 there exist a sequence of points in supp( P B ) that converge to z . Let { e k } K k =1 be K code vectors 759 independently generated from P B . Then the empirical distribution of { e k } K k =1 tends to P B as the 760 size K tends to infinity. Since Ω = supp( P A ) is a bounded region, we have the following: 761

<!-- formula-not-decoded -->

This quantity is an upper bound on the quantization error E ( { z i } ; { e k } ) . Thus, 762

<!-- formula-not-decoded -->

This demonstrates that P B has vanishing quantization error asymptotically. Furthermore, for any K 763 code vectors { e k } K k =1 independently drawn from P B , we have { e k } K k =1 ⊆ Ω . Since the empirical 764 distribution of { z i } N i =1 tends to P A as the feature sample size N tends to infinity, we can easily show 765 that for any fixed { e k } K k =1 ⊆ Ω , the codebook utility rate satisfies 766

<!-- formula-not-decoded -->

This shows that { e k } K k =1 attains full utilization asymptotically, and thus P B attains full utilization asymptotically.

On the other hand, we assume P B attains full utilization and vanishing quantization error asymptotically. Then we first claim that supp( P A ) ⊆ supp( P B ) . Since P B has vanishing quantization error asymptotically, then for any z ∈ supp( P A ) , there exist a sequence of points in supp( P B ) that converge to z . This implies that supp( P A ) ⊆ supp( P B ) and thus supp( P A ) ⊆ supp( P B ) .

To show supp( P B ) = supp( P A ) , it remains to show supp( P B ) ⊆ supp( P A ) . In fact, if supp( P B ) ⊆ supp( P A ) does not hold, then there exists an open region R⊆ supp( P B ) -supp( P A ) such that P B ( R ) &gt; 0 and

<!-- formula-not-decoded -->

for some ϵ 0 &gt; 0 . Since supp( P A ) ⊆ supp( P B ) , then there exists a sufficiently large K 0 such that 776 the event 777

<!-- formula-not-decoded -->

has some positive probability C &gt; 0 . Then with a positive probability of at least C · P B ( R ) , we can 778 pick the first K 0 code vectors from Equation (7) and the ( K 0 +1) th code vector from R . For any 779 such codebook of size K 0 +1 , we know the ( K 0 +1) th code vector will never be used regardless of 780 the choice of the feature set { z i } . Therefore, the codebook utilization 781

<!-- formula-not-decoded -->

This contradicts the property that P B attains full utilization asymptotically. Thus, supp( P B ) ⊆ 782 supp( P A ) must hold. This concludes the proof. 783

## B Statistical Distances over Gaussian Distributions 784

We first introduce the definition of Wasserstein distance. 785

Definition 4. The Wasserstein distance or earth-mover distance with p norm is defined as below: 786

<!-- formula-not-decoded -->

767

768

769

770

771

772

773

774

775

where Π( P r , P g ) denotes the set of all joint distributions γ ( x, y ) whose marginals are P r and P g 787 respectively. Intuitively, when viewing each distribution as a unit amount of earth/soil, the Wasserstein 788 distance (also known as earth-mover distance) represents the minimum cost of transporting 'mass' 789 from x to y to transform distribution P r into distribution P g . When p = 2 , this is called the quadratic 790 Wasserstein distance. 791

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

In this paper, we achieve distributional matching using the quadratic Wasserstein distance under Gaussian distribution assumptions. We also examine other statistical distribution distances as potential loss functions for distributional matching and compare them with the Wasserstein distance. Specifically, we provide the Kullback-Leibler divergence and the Bhattacharyya distance over Gaussian distributions in Lemma 5 and Lemma 6. It can be observed that the KL divergence for two Gaussian distributions involves calculating the determinant of covariance matrices, which is computationally expensive in moderate and high dimensions. Moreover, the calculation of the determinant is sensitive to perturbations and it requires full rank (In the case of not full rank, the determinant is zero, rendering the logarithm of zero undefined), which can be impractical in many cases. Other statistical distances like Bhattacharyya Distance suffer from the same issue. In contrast, quadratic Wasserstein distance does not require the calculation of the determinant and full-rank covariance matrices.

803

804

805

806

807

808

Lemma 5 (Kullback-Leibler divergence [22]) . Suppose two random variables Z 1 ∼ N ( µ 1 , Σ 1 ) and Z 2 ∼ N ( µ 2 , Σ 2 ) obey multivariate normal distributions, then Kullback-Leibler divergence between Z 1 and Z 2 is:

<!-- formula-not-decoded -->

Lemma 6 (Bhattacharyya Distance [4]) . Suppose two random variables Z 1 ∼ N ( µ 1 , Σ 1 ) and Z 2 ∼ N ( µ 2 , Σ 2 ) obey multivariate normal distributions, Σ = 1 2 ( Σ 1 + Σ 2 ) , then bhattacharyya distance between Z 1 and Z 2 is:

<!-- formula-not-decoded -->

## C Understanding Codebook Collapse Through the Lens of Voronoi Partition 809

Figure 8: Visualization of the Voronoi partition. The symbols · and × represent the feature and code vectors, respectively.

<!-- image -->

## C.1 The Definition of Voronoi Partition and Its Connection to Codebook Collapse

Let X be a metric space with distance function d ( · , · ) , and given a set of code vectors { e k } K k =1 . The Voronoi cell, or Voronoi region, R k , associated with the code vector e k is the set of all points in X whose distance to e k is not greater than their distance to the other code vectors e j , where j is any index different from k . Mathematically, this can be expressed as:

̸

<!-- formula-not-decoded -->

The Voronoi diagram is simply the tuple of cells {R k } K k =1 . As depicted in Figure 8, there are 12 code vectors which partition the metric space into 12 regions according to the R k . When d is a a distance function based on the ℓ 2 norm, the vector quantization (VQ) process can be equivalently understood through the regions R k as:

<!-- formula-not-decoded -->

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

Where z i is an arbitrary feature vector. Equation 10 offers an alternative approach for nearest neighbor search in code vector selection. Specifically, this involves first identifying the partition region R j to which the feature vector z i belongs, and then directly obtaining the nearest code vector e j based on the region's id j .

Relation to Codebook Collapse The most severe case of codebook collapse occurs when all feature vectors belong to the same partition region. As illustrated in Figure 8(a), all feature vectors are confined to a single partition region in the upper right corner, resulting in the utilization of only one code vector. To prevent codebook collapse, it is crucial for feature vectors to be distributed across all partition regions as evenly as possible, as depicted in Figure 8(d).

## C.2 Why Existing Vector Quantization Strategies Fail to Address Codebook Collapse

This section offers an in-depth analysis of why existing VQ methods inherently struggle to address codebook collapse. We use Vanilla VQ [36] and VQ methods based on the k -means algorithm [29] as illustrative examples. These two approaches share similarities in their assignment step but differ in their update mechanisms.

Assignment Step Suppose there is a set of feature vectors { z i } N i =1 and code vectors { e k } K k =1 . In the t -th assignment step, both algorithms partition the feature space into Voronoi cells, based on which we assign all feature vectors to their nearest code vectors as follows:

̸

<!-- formula-not-decoded -->

Update Step in Vanilla VQ It updates the code vectors using gradient descent through the loss function provided below.

<!-- formula-not-decoded -->

Update Step in k -means-based VQ It updates the code vectors by using an exponential moving average of the feature vectors assigned to each code vector:

<!-- formula-not-decoded -->

Codebook Collapse in Two VQs While both VQ methods employ different update strategies for the code vectors, they still suffer from codebook collapse. This is because, in the assignment step, the learnable Voronoi partition does not guarantee that all Voronoi cells will be assigned feature vectors, as illustrated in Figures 8(a) to 8(c). Especially when the codebook size is large, there are more Voronoi cells, and inevitably, some cells remain unassigned. In such cases, the corresponding code vectors remain unupdated and underutilized.

Connection to Distribution Matching and Solutions In Appendix H, we demonstrated through synthetic experiments that the effectiveness of both VQ methods heavily relies on the codebook initialization. Only when the codebook distribution is initialized to approximate the feature distribution can codebook collapse be effectively mitigated. However, in practical applications, the feature distribution is often unknown and evolves dynamically during training. To address this issue, we propose an explicit distributional matching constraint that ensures the codebook distribution aligns closely with the feature distribution, thereby achieving 100% codebook utilization.

## D Complementary Roles of Criterion 2 and 3 in Assessing Codebook Collapse

To explain the complementary roles of Criterion 2 and 3 (defined in Section 2.2), we provide visual elucidations for enhanced clarity and understanding. The metric U is capable of quantifying the completeness of codebook utilization. As depicted in the Figure 9(a) and 9(b), the values of U are

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

Figure 9: Visualization of the evaluation criteria ( U , C ) .

<!-- image -->

50% and 100% , respectively 2 . However, U alone is insufficient to evaluate the degree of codebook collapse, as it fails to address the scenario depicted in Figure 9(c). Although all code vectors are utilized, the code vector e 3 excessively dominates the codebook utilization, resulting in an extreme imbalance. This imbalanced codebook utilization can be considered a form of codebook collapse, despite U reaching its maximum value. This observation motivates the proposal of Criterion 3, which is capable of gauging the imbalance or uniformity inherent in codebook utilization.

When compared in Figure 9(b) and 9(c), the value of C are 10 . 00 and 1 . 02 , respectively, demonstrating that Criterion 3 is capable of distinguishing the imbalance of code vector utilization p k under conditions where cases share the same U , e.g., U = 100% . Additionally, Criterion 3 categorizes Figure 9(c) as indicative of codebook collapse, as the value C nearly reaches its minimum of 1 . 0 , a result that resonates with our desired interpretation. However, it is essential to note that Criterion 3 alone does not suffice to evaluate the degree of codebook collapse. When scrutinizing Figure 9(a) and 9(d), despite the identical C , there exists a stark disparity in U . This observation underscores that the value of C is inadequate for quantifying the proportion of actively utilized code vectors.

In this paper, we adopt the combination of Criterion 2 and 3 to quantitatively assess the extent of codebook collapse. A robust mitigation of codebook collapse is indicated solely when both U and C exhibit substantial values.

## E Interpretation of Qualitative Distributional Matching Results in Figure 3

This section interprets the experimental results presented in Figure 3. The VQ process relies on nearest neighbor search for code vector selection. As evident from Figure 3(a) to 3(d), actively selected code vectors are predominantly those located in close proximity to or within the feature distribution, while distant ones remain unselected. This leads to highly uneven code vector utilization p k , with those closer to the feature distribution being excessively used. This elucidates the significantly low U and C observed in Figure 3(a). Furthermore, a notable quantization error, e.g., E = 1 . 19 in Figure 3(a), arises when the codebook and feature distributions are mismatched, forcing feature vectors outside the codebook to settle for distant code vectors. Conversely, as the disk centers align, leading to a closer match between the two distributions, an increased number of code vectors become actively engaged. Additionally, code vectors are utilized more uniformly, and feature vectors can select nearer counterparts. This accounts for the improvement of criterion triple values towards optimality as the distributions align.

Analogously, we can employ nearest neighbor search to interpret the second case. When code vectors are distributed within the range of feature vectors, as illustrated in Figure 3(e) and Figure 3(f), the majority of code vectors would be actively utilized, ensuring high U . However, the utilization of these code vectors is not uniform; code vectors on the periphery of the codebook distribution are more frequently used, leading to relatively low C . Feature vectors on the periphery will have larger distances to their nearest code vectors, resulting in higher E . Conversely, when feature vectors fall within the range of code vectors, as depicted in Figure 3(g) and Figure 3(h), outer code vectors remain largely unused, leading to a lower U and C . Since only inner code vectors are active, each feature vector can find a nearby counterpart, maintaining low E .

2 This discrepancy arises because, in Figure 9(a) only half of code vectors' utilization p k (as defined in Criterion 3) exceeds zero, whereas in Figure 9(b), the utilization p k of of all code vectors surpasses zero.

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

Figure 10: Quantitative analyses of the criterion triple when P A and P B are Gaussian distributions.

<!-- image -->

## F Supplementary Quantitative Analyses on Distribution Matching: Further Supporting the Main Findings in Section 2.3

To further elucidate the effects of the distributional matching, we conduct more quantitative analyses centered around the criterion triple ( E , U , C ) .

## F.1 Codebook Distribution and Feature Distribution are Gaussian Distributions

We begin by assuming that the distributions P A and P B are Gaussian. We generate a set of feature vectors { z i } N i =1 from N d ( 0 , I ) and a set of code vectors { e k } K k =1 from N d ( µ · 1 , I ) 3 , with µ varying within { 0 . 0 , 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 , 2 . 5 } . The criterion triple results are presented in Figures 10(a) to 10(c), Figures 10(g) to 10(i), and Figures 10(m) to 10(o). Across all tested configurations of K,d,N , we consistently observe that when µ = 0 -indicating identical distributions between P A and P B -the criterion triple achieves the lowest E , highest U , and largest C . This empirical evidence reinforces the effectiveness of aligning feature and codebook distributions in VQ.

Additionally, we further analyze the criterion triple by varying the covariance matrix. We sample a set of feature vectors { z i } N i =1 from the distribution N d ( 0 , I ) and a set of code vectors { e k } K k =1 from N d ( 0 , σ 2 I ) , where σ is selected from { 1 , 2 , 3 , 4 , 5 , 6 } . The results for the criterion triple are shown in Figures 10(d) to 10(f), Figures 10(j) to 10(l), and Figures 10(p) to 10(r). When σ = 1 , indicating identical distributions between P A and P B , all three evaluation criteria reach their optimal values: the lowest E , highest U , and largest C across all tested values of K,d,N . This result corroborates our earlier findings.

## F.2 Codebook Distribution and Feature Distribution are Unifrom Distributions

The above conclusion holds when P A and P B are other types of distributions, such as the uniform distribution. As shown in Figure 11, we sample a set of feature vectors { z i } N i =1 from the distribution Unif d ( -1 , 1) and a set of code vectors { e k } K k =1 from Unif d ( ν -1 , ν +1) , where ν is selected from the set { 0 . 0 , 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 , 2 . 5 } or from Unif d ( -ζ, ζ ) , with ζ drawn from the set { 1 , 2 , 3 , 4 , 5 , 6 } . We observe that when µ = 0 or ζ = 1 -indicating that P A and P B have identical distributions-the performance in terms of the criterion triple is optimal, achieving the lowerest E , the highest U , and the largest C across all tested values of K,d,N . Therefore, we conclude that our quantitative analyses are distribution-agnostic and can be generalized to other distributions.

3 1 represents the vector of all ones.

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

Figure 11: Quantitative analyses of the criterion triple when P A and P B are uniform distributions.

<!-- image -->

## G The Significant Impact of Distribution Variance on Quantization Error

As discussed in Section 2.3 and 2.4, the optimal criterion triple is achieved when the distributions P A and P B are identical. In this section, we further analyze the criterion triple by the lens of distribution variance under the condition that both distributions are identical. Specifically, we first sample a set of feature vectors { z i } N i =1 along with a set of code vectors { e k } K k =1 from the distribution N d ( 0 , σ 2 I ) or the distribution Unif d ( -ζ, ζ ) . We then calculate the evaluation criteria according to their definitions in Section 2.2. As demonstrated in Table 4, σ and ζ have a substantial impact on E , while U and C remains largely unaffected.

Table 4: The criterion triple influence by the distribution variance.

| Evaluation Criteria   |           σ |           σ |           σ |         σ |         σ |           ζ |           ζ |           ζ |          ζ |         ζ |
|-----------------------|-------------|-------------|-------------|-----------|-----------|-------------|-------------|-------------|------------|-----------|
| Evaluation Criteria   |    0.0001   |    0.001    |    0.01     |    0.1    |    1      |    0.0001   |    0.001    |    0.01     |    0.1     |    1      |
| E                     |    1.25e-08 |    1.25e-06 |    0.000125 |    0.0124 |    1.25   |    3.27e-09 |    3.27e-07 |    3.27e-05 |    0.00327 |    0.327  |
| U                     |    0.9934   |    0.9938   |    0.994    |    0.9934 |    0.9941 |    0.9993   |    0.9986   |    0.999    |    0.9992  |    0.9989 |
| C                     | 7265.3      | 7260.3      | 7267.7      | 7255      | 7275.8    | 7380.2      | 7372.2      | 7387.9      | 7397.5     | 7391.6    |

This experimental finding suggests that when the distribution variance of the feature vectors is uncontrollable or unknown, reporting a comparison of quantization error among various VQ methods is unreasonable. This is because the improvement in quantization error is predominantly attributed to the reduction in distribution variance rather than the effectiveness of the VQ methods. To evaluate various VQ methods in terms of the criterion triple, we establish an atomic and fair experimental setting in Appendix H, where the feature distributions for all VQ methods are identical.

## H A Fair Setting to Evaluate Criterion Triple Evaluation

The distribution variance has a substantial impact on E , as detailed in Appendix G. Therefore, the comparison of the quantization error among various VQ methods is unreasonable when the variance of the feature vectors is uncontrollable or unknown. This is because any improvement in quantization error is primarily attributed to the variance reduction rather than the inherent effectiveness of the VQ methods. To ensure a fair criterion triple evaluation, we provide a controlled experimental setting.

Specifically, we fix the feature distributions for all VQ methods to the same Gaussian distributions by setting z i ∼ N d ( µ · 1 , I ) . Additionally, we initialize the codebook distribution as the standard Gaussian distribution across all VQ methods by sampling e k ∼ N d ( 0 , I ) . In this experimental setup, the distribution variance is controlled to be the same for all VQ methods.

Our baseline includes Vanilla VQ [36], EMA VQ [29], Online VQ [42], and Linear VQ (a linear layer projection for frozen code vectors) [43, 44]. In all VQ algorithms, we treat the sampled code vectors

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

Figure 12: The performance metrics ( E , U , C ) for various VQ approaches. For panels (a) to (d), the codebook distribution is initialized as a Gaussian distribution, while for panels (e) to (h), the codebook distribution is initialized as a uniform distribution.

<!-- image -->

as trainable parameters and optimize them using the respective algorithms. See detailed experimental specifications in Appendix I.4.

As visualized in Figures 12(a) to 12(c), Wasserstein VQ outperforms all baselines in terms of the criterion triple ( E , U , C ) , especially when the feature distribution and the initialized codebook distribution have large deviations. Although existing VQ methods can perform well with µ = 0 , this scenario is impractical, as the feature distribution is unknown and changes dynamically during training. When there is a large initial distribution gap between the codebook and the features (at µ = 5 ), all existing VQ methods perform poorly. This indicates that the effectiveness of existing VQ methods is heavily dependent on their codebook initialization.

As observed in Figure 12(d), the Wasserstein distance of all existing VQ methods is obviously larger compared to that of Wasserstein VQ when µ ≥ 1 . 0 , indicating that existing methods cannot achieve effective distribution alignment between features and the codebook. Conversely, Wasserstein VQ eliminates the reliance on codebook initialization via proposed explicit distributional matching regularization, thereby delivering the best performance in criterion metrics.

We can arrive at the same conclusion when the feature and codebook distributions are uniform, in which feature vectors are generated from Unif d ( ν -1 , ν +1) and code vectors are initialized from Unif d ( -1 , 1) . As shown in Figures 12(e) to 12(h), Wasserstein VQ performs the best. This suggests that, despite being based on Gaussian assumptions, the effectiveness of our method exhibits a certain degree of distribution-agnostic behavior.

## I The Details of Synthetic Experiments

## I.1 Experimental Details in Section 2.3

As depicted in Figure 3 in Section 2.3, we conduct a qualitative analyses of the criterion triple. Specifically, we sample a set of feature vectors { z i } N i =1 from within the red circle, and a collection of code vectors { e k } K k =1 from within the green circle, with parameters set to K = 400 , N = 10000 and d = 2 for the calculation of the criterion triple ( E , U , C ) . For the visualization, we select 10% of the feature vectors and 90% of the code vectors for plotting.

## I.2 Experimental Details in Appendix F

As illustrate in Figure 10 in Appendix F.1, we undertake comprehensive quantitative analyses centered around the criterion triple ( E , U , C ) . In these analyses, we assume that P A and P B are Gaussian distributions, from which we sample a set of feature vectors { z i } N i =1 and a collection of code vectors

980

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

1007

1008

1009

1010

1011

1012

1013

Table 5: Hyperparameters for the experiments in Table 1, 2, 3, and Table 6 and 7.

| Frameworks                                                                                                                                                                                                                                                     | VQ-VAE                                                                                     | VQ-VAE                                                                                       | VQ-VAE                                                                                                  | VQGAN                                                                                                    |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Dataset Input size Latent size encoder/decoder channels encoder/decoder channel mult. Batch size Initial Learning rate lr Codebook Loss Coefficient Perceptual loss Coefficient Adversarial loss Coefficient Codebook dimensions Training Epochs GPU Resources | CIFAR-10/SVHN 32 × 32 × 3 8 × 8 × 8 64 [1 , 1 , 2] 128 5 × 10 - 5 1 . 0 0 0 8 50 V100 16GB | FFHQ 256 × 256 × 3 16 × 16 × 8 64 [1 , 1 , 2 , 2 , 4] 32 5 × 10 - 5 1 . 0 0 0 8 30 A100 40GB | ImageNet 256 × 256 × 3 16 × 16 × 8 64 [1 , 1 , 2 , 2 , 4] 32 5 × 10 - 5 1 . 0 0 . 0 0 . 0 8 4 A100 40GB | FFHQ 256 × 256 × 3 16 × 16 × 8 128 [1 , 1 , 2 , 2 , 4] 64 5 × 10 - 4 1 . 0 1 . 0 0 . 2 8 200 2 A100 80GB |

{ e k } K k =1 . The default parameters are set to N = 200 , 000 , K = 1024 , and d = 32 for all figures unless otherwise specified. For instance, in Figure 10(a), N and d are taken at their default values, while the K is varied within the set { 128 , 256 , 512 , 1024 , 2048 , 4096 , 8192 , 16284 } . Additionally, each synthetic experiment is repeated five times, and the average results are reported, along with the calculation of 95% confidence intervals. In all figures, mean results are represented by points, while the confidence intervals are shown as shaded areas. Identical parameter settings are employed when P A and P B are uniform distributions, as illustrated in Figure 11 in Appendix F.2.

## I.3 Experimental Details in Appendix G

We set K = 8192 , d = 8 , N = 100000 when calculating the criterion triple ( E , U , C ) in Appendix G. Each synthetic experiment is repeated five times, and the average results are reported in Table 4.

## I.4 Experimental Details in Appendix H

We provide experimental details of Figure 12 in Appendix H. In our experimental setup, we evaluate five distinct VQ algorithms using the criterion triple ( E , U , C ) . All experiments run on a single NVIDIA A100 GPU, with a codebook size K of 16,384 and dimensionality d of 8 across all algorithms. Each algorithm trains for 2,000 steps, with 50,000 feature vectors sampled from the specified Gaussian distribution at each step. For Wasserstein VQ , Vanilla VQ, and VQ + MLP, we use the SGD optimizer for training. For VQ EMA and Online Clustering, we use classical clustering algorithmsk -means [5] and k -means++[1]-to update code vectors.

## J Experimental Details in Section 4

Data Augmentation For FFHQ and ImageNet-1k datasets, we follow LLama Gen [33] and apply iterative box downsampling to resize images to 256×256 resolution. For CIFAR-10 and SVHN, the images are kept at their original resolution. Details are provided in Table 5.

Encoder-Decoder Architecture For the ImageNet and FFHQ datasets, within both the VQ-VAE and VQGAN frameworks, our proposed Wasserstein VQ and all baseline methods adopt identical encoder-decoder architectures and parameter configurations, following the original VQGAN implementation [9]. Across all baselines in these frameworks, the encoder-a U-Net [31]-downscales the input image by a factor of 16. For CIFAR-10 and SVHN datasets, the encoder reduces the input resolution by a factor of 4. Further details are provided in Table 5.

Training Details All experiments employ identical training settings: we use the AdamW optimizer [23] with β 1 = 0 . 9 and β 1 = 0 . 95 , an initial learning rate lr , and apply a half-cycle cosine decay schedule following a linear warm-up phase. For specific details on training epochs and batch sizes, refer to Table 5.

Loss Weight For all three baselines, β is typically set to a value within the range [0 . 25 , 2] . In our experiments, β is set to a fixed value of 1 . 0 . For our proposed Wasserstein VQ model, we set

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

Table 6: Comparison of VQ-VAEs trained on CIFAR-10 dataset following [36].

| Approach       |   Tokens |   Codebook Size | U ( ↑ )   |   C ( ↑ ) |   PSNR( ↑ ) |   SSIM( ↑ ) |   Rec. Loss ( ↓ ) |
|----------------|----------|-----------------|-----------|-----------|-------------|-------------|-------------------|
| Vanilla VQ     |       64 |            8192 | 2.7%      |     186.9 |       27.15 |        0.83 |            0.0147 |
| EMA VQ         |       64 |            8192 | 99.7%     |    6416.1 |       29.43 |        0.88 |            0.0095 |
| Online VQ      |       64 |            8192 | 22.1%     |     995.4 |       28.2  |        0.85 |            0.0123 |
| Wasserstein VQ |       64 |            8192 | 100.0%    |    7781.8 |       29.88 |        0.9  |            0.0085 |
| Vanilla VQ     |       64 |           16384 | 1.6%      |     220.3 |       27.36 |        0.84 |            0.0141 |
| EMA VQ         |       64 |           16384 | 80.8%     |   10557.3 |       29.43 |        0.88 |            0.0093 |
| Online VQ      |       64 |           16384 | 13.4%     |     798.5 |       27.54 |        0.82 |            0.0141 |
| Wasserstein VQ |       64 |           16384 | 100.0%    |   15583.7 |       30.19 |        0.9  |            0.008  |
| Vanilla VQ     |       64 |           32768 | 0.5%      |     154.8 |       27.1  |        0.83 |            0.015  |
| EMA VQ         |       64 |           32768 | 54.4%     |   14427   |       29.57 |        0.88 |            0.0091 |
| Online VQ      |       64 |           32768 | 7.2%      |    1556   |       28.84 |        0.87 |            0.0106 |
| Wasserstein VQ |       64 |           32768 | 99.0%     |   29845.1 |       30.63 |        0.91 |            0.0071 |

Table 7: Comparison of VQ-VAEs trained on SVHN dataset following [36].

| Approach       |   Tokens |   Codebook Size | U ( ↑ )   |   C ( ↑ ) |   PSNR( ↑ ) |   SSIM( ↑ ) |   Rec. Loss ( ↓ ) |
|----------------|----------|-----------------|-----------|-----------|-------------|-------------|-------------------|
| Vanilla VQ     |       64 |            8192 | 8.1%      |     533.1 |       37.81 |        0.97 |            0.0018 |
| EMA VQ         |       64 |            8192 | 56.8%     |    3363   |       40.38 |        0.98 |            0.001  |
| Online VQ      |       64 |            8192 | 27.8%     |    1325.1 |       39.04 |        0.97 |            0.0016 |
| Wasserstein VQ |       64 |            8192 | 88.2%     |    6154.5 |       41.04 |        0.98 |            0.0009 |
| Vanilla VQ     |       64 |           16384 | 3.4%      |     446   |       37.87 |        0.97 |            0.0017 |
| EMA VQ         |       64 |           16384 | 22.2%     |    2593.8 |       40.19 |        0.98 |            0.0011 |
| Online VQ      |       64 |           16384 | 13.5%     |    1090.5 |       39.12 |        0.97 |            0.0014 |
| Wasserstein VQ |       64 |           16384 | 87.5%     |   11967.2 |       41.49 |        0.98 |            0.0008 |
| Vanilla VQ     |       64 |           32768 | 1.8%      |     467.5 |       37.87 |        0.97 |            0.0017 |
| EMA VQ         |       64 |           32768 | 35.8%     |    7662.9 |       40.25 |        0.98 |            0.001  |
| Online VQ      |       64 |           32768 | 7.0%      |    1334.8 |       39.26 |        0.97 |            0.0014 |
| Wasserstein VQ |       64 |           32768 | 88.7%     |   24376.3 |       41.84 |        0.98 |            0.0008 |

β to a much smaller value, e.g., β = 0 . 1 for VQ-VAE and VQGAN. The smaller β values enable the Wasserstein distance to dominate the loss function, thereby more effectively narrowing the gap between the distributions.

## K VQ-VAE Performance on CIFAR-10 and SVHN datasets

Due to space limitations in the main text, we have relocated the VQ-V AE evaluation on CIFAR-10 and SVHN datasets to the appendix. As demonstrated in Table 6 and 7, our Wasserstein VQ consistently outperforms all baselines across both datasets, achieving superior results on nearly all evaluation metrics regardless of codebook size. Notably, we observe that Wasserstein VQ fails to reach 100% codebook utilization on SVHN, which may be attributed to the dataset's limited diversity.

## L Analyses on Codebook Size and Dimensionality

We investigate the impact of the codebook size K on the performance of VQ by varying across a wide range: K ∈ [1024 , 2048 , 4096 , 8192 , 16384 , 50000 , 100000] . As shown in Table 1 and Table 8, the vanilla VQ model suffers from severe codebook collapse even with a relatively small K , such as K = 1024 . In contrast, improved algorithms like EMA VQ and Online VQ can handle smaller codebook sizes effectively, but they still experience codebook collapse when K is very large, e.g., K ≥ 50000 . Notably, the Wasserstein VQ model consistently maintains 100% codebook utilization, irrespective of the codebook size. This underscores the effectiveness of distributional matching via the quadratic Wasserstein distance in mitigating the issue of codebook collapse.

We further investigate the impact of codebook dimensionality d on VQ performance. Conducting 1032 experiments on CIFAR-10 with dimensionality d ranging from 2 to 32, our proposed Wasserstein VQ 1033 consistently outperforms all baselines regardless of dimensionality, as shown in Table 9. Notably, we 1034 observe the curse of dimensionality phenomenon-performance degrades as dimensionality increases. 1035 Vanilla VQ exhibits the most severe degradation, followed by EMA VQ and Online VQ, while our 1036 Wasserstein VQ shows only minimal codebook utilization reduction. 1037

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

Table 8: Supplementary comparison of VQ-VAEs trained on FFHQ dataset following [36] w.r.t codebook size

K

.

| Approach       |   Tokens |   Codebook Size | U ( ↑ )   |   C ( ↑ ) |   PSNR( ↑ ) |   SSIM( ↑ ) |   Rec. Loss ( ↓ ) |
|----------------|----------|-----------------|-----------|-----------|-------------|-------------|-------------------|
| Vanilla VQ     |      256 |            1024 | 51.7%     |     446.2 |       27.64 |        73   |            0.0125 |
| EMA VQ         |      256 |            1024 | 74.1%     |     618.9 |       27.66 |        72.7 |            0.0125 |
| Online VQ      |      256 |            1024 | 100.0%    |     759.3 |       28.08 |        74   |            0.0114 |
| Wasserstein VQ |      256 |            1024 | 100.0%    |     977.4 |       28.11 |        74.4 |            0.0112 |
| Vanilla VQ     |      256 |            2048 | 27.6%     |     453   |       27.78 |        73.8 |            0.0121 |
| EMA VQ         |      256 |            2048 | 100%      |    1608   |       28.39 |        74.9 |            0.0107 |
| Online VQ      |      256 |            2048 | 100%      |    1462.6 |       28.34 |        74.6 |            0.0108 |
| Wasserstein VQ |      256 |            2048 | 100%      |    1840.5 |       28.32 |        75.3 |            0.0107 |
| Vanilla VQ     |      256 |            4096 | 12.5%     |     435   |       27.84 |        73.7 |            0.0119 |
| EMA VQ         |      256 |            4096 | 76.7%     |    2443.1 |       28.49 |        75   |            0.0104 |
| Online VQ      |      256 |            4096 | 70.7%     |    1600   |       28.25 |        74.1 |            0.011  |
| Wasserstein VQ |      256 |            4096 | 100%      |    3895.4 |       28.54 |        75.1 |            0.0102 |
| Vanilla VQ     |      256 |            8192 | 5.6%      |     398.1 |       27.69 |        73.5 |            0.0122 |
| EMA VQ         |      256 |            8192 | 28.9%     |    1839.2 |       28.39 |        74.8 |            0.0106 |
| Online VQ      |      256 |            8192 | 34.9%     |    1474.4 |       28.15 |        73.9 |            0.0113 |
| Wasserstein VQ |      256 |            8192 | 100%      |    7731.5 |       28.81 |        76.2 |            0.0099 |

Table 9: Analysis On codebook dimension by the comparison of VQ-VAEs trained on CIFAR-10 dataset following [36]. (The codebook size K is fixed to 16384)

| Approach       |   Tokens |   Codebook Dim | U ( ↑ )   |   C ( ↑ ) |   PSNR( ↑ ) |   SSIM( ↑ ) |   Rec. Loss ( ↓ ) |
|----------------|----------|----------------|-----------|-----------|-------------|-------------|-------------------|
| Vanilla VQ     |      256 |              2 | 3.8%      |    532.2  |       27    |        0.8  |            0.0162 |
| EMA VQ         |      256 |              2 | 97.6%     |  14460.3  |       27.25 |        0.8  |            0.0155 |
| Online VQ      |      256 |              2 | 9.0%      |    611.8  |       26.62 |        0.79 |            0.0178 |
| Wasserstein VQ |      256 |              2 | 99.3%     |  12278.9  |       27.3  |        0.81 |            0.0155 |
| Vanilla VQ     |      256 |              4 | 1.3%      |    176.7  |       27.15 |        0.83 |            0.0149 |
| EMA VQ         |      256 |              4 | 99.8%     |  13153.9  |       29.57 |        0.89 |            0.0092 |
| Online VQ      |      256 |              4 | 11.1%     |    877.7  |       26.69 |        0.79 |            0.0173 |
| Wasserstein VQ |      256 |              4 | 100.0%    |  15724.7  |       29.93 |        0.89 |            0.0087 |
| Vanilla VQ     |      256 |              8 | 1.6%      |    220.3  |       27.36 |        0.84 |            0.0141 |
| EMA VQ         |      256 |              8 | 80.8%     |  10557.3  |       29.43 |        0.88 |            0.0009 |
| Online VQ      |      256 |              8 | 13.4%     |    798.5  |       27.54 |        0.82 |            0.0141 |
| Wasserstein VQ |      256 |              8 | 100.0%    |  15583.7  |       30.19 |        0.9  |            0.008  |
| Vanilla VQ     |      256 |             16 | 1.1%      |    150.8  |       27.05 |        0.83 |            0.0152 |
| EMA VQ         |      256 |             16 | 32.5%     |   4169.2  |       29.31 |        0.88 |            0.0099 |
| Online VQ      |      256 |             16 | 18.2%     |   2051    |       28.29 |        0.85 |            0.0122 |
| Wasserstein VQ |      256 |             16 | 99.2%     |  14832.2  |       30.27 |        0.91 |            0.0078 |
| Vanilla VQ     |      256 |             32 | 0.7%      |     94.37 |       26.67 |        0.81 |            0.0165 |
| EMA VQ         |      256 |             32 | 7.0%      |    942.7  |       28.24 |        0.85 |            0.0122 |
| Online VQ      |      256 |             32 | 18.8%     |   2278    |       28.92 |        0.87 |            0.0104 |
| Wasserstein VQ |      256 |             32 | 96.4%     |  14056.9  |       30.39 |        0.91 |            0.0076 |

## M Discussion with VQ-WAE [37]

VQ-WAE [37] introduces an alternative approach to distributional matching by employing Optimal Transport to optimize codebook vectors. Compared with our proposed distributional matching method, there are three key differences.

First, regarding theoretical contributions : VQ-WAE [37] claims that achieving optimal transport (OT) between code vectors and feature vectors yields the best reconstruction performance. Their notion of optimality encompasses both the VQ process and the encoder-decoder reconstruction pipeline. While we contend that incorporating complex encoder-decoder functions renders rigorous theoretical analysis intractable, VQ-WAE nevertheless asserts this conclusion. In contrast, our work deliberately excludes encoder-decoder components, focusing solely on the VQ process, which admits rigorous mathematical modeling. Through our proposed criterion triple, we theoretically prove that distributional matching guarantees optimal performance.

Second, regarding distribution modeling : VQ-WAE [37] assumes both code vectors and feature vectors follow uniform discrete distributions, whereas our method models them as continuous distributions. Specifically, VQ-WAE [37] represents the distributions of feature vectors { z i } N i =1 and

Table 10: Reconstruction performance ( ↓ : the lower the better and ↑ : the higher the better). † :Results cited from VQ-WAE [37]. Codebook size K is fixed to 512 .

| Dataset   | Model            |   Tokens |   SSIM ↑ |   PSNR ↑ |   LPIPS ↓ | Rec. Loss ( ↓ )   |   Perplexity ↑ |
|-----------|------------------|----------|----------|----------|-----------|-------------------|----------------|
| CIFAR10   | VQ-VAE †         |       64 |       70 |    23.14 |      0.35 |                   |           69.8 |
|           | SQ-VAE †         |       64 |       80 |    26.11 |      0.23 |                   |          434.8 |
|           | VQ-WAE †         |       64 |       80 |    25.93 |      0.23 |                   |          497.3 |
|           | VQ-WAE (Our run) |       64 |       13 |    14.6  |      0.41 | 0.247             |            1   |
|           | Vanilla VQ       |       64 |       83 |    27.19 |      0.03 | 0.015             |          192.5 |
|           | EMA VQ           |       64 |       84 |    27.97 |      0.04 | 0.013             |          436.1 |
|           | Online VQ        |       64 |       84 |    27.87 |      0.04 | 0.013             |          451.4 |
|           | Wasserstein VQ   |       64 |       86 |    28.26 |      0.03 | 0.012             |          481.7 |
| SVHN      | VQ-VAE †         |       64 |       88 |    26.94 |      0.17 |                   |          114.6 |
|           | SQ-VAE †         |       64 |       96 |    35.37 |      0.06 |                   |          389.8 |
|           | VQ-WAE †         |       64 |       96 |    34.62 |      0.07 |                   |          485.1 |
|           | VQ-WAE (Our run) |       64 |       25 |    15.87 |      0.26 | 0.2026            |            1   |
|           | Vanilla VQ       |       64 |       97 |    38.18 |      0.01 | 0.0016            |          407.1 |
|           | EMA VQ           |       64 |       97 |    38.35 |      0.01 | 0.0017            |          408.9 |
|           | Online VQ        |       64 |       97 |    38.54 |      0.01 | 0.0017            |          421.5 |
|           | Wasserstein VQ   |       64 |       97 |    38.25 |      0.01 | 0.0016            |          423.5 |

code vectors { e k } K k =1 as empirical measures: 1053

<!-- formula-not-decoded -->

where δ z i and δ e k denote Dirac delta functions centered at z i and e k , respectively. To align P A and 1054 P B , VQ-WAE formulates the OT problem as: 1055

<!-- formula-not-decoded -->

where P is the transport plan, and the feasible set is: 1056

<!-- formula-not-decoded -->

1057

1058

In contrast, we simplify the distributional assumption by modeling P A and P B as Gaussian distributions.

Third, regarding computational efficiency , The OT problem in VQ-WAE is prohibitively complex, 1059 whereas our quadratic Wasserstein distance incurs minimal overhead. To mitigate complexity, VQ1060 WAE employs a Kantorovich potential network. However, upon reproducing their code (no official 1061 implementation was released; we derived it from their ICLR 2023 supplementary material 4 ), we 1062 observed severe non-convergence-the method degenerated to using a single code vector, failing 1063 to achieve distributional matching. Notably, VQ-WAE underperformed all other VQ baselines 1064 (Table 10). 1065

In comparison, our quadratic Wasserstein distance (Equation 4) requires only low-dimensional matrix 1066 operations (e.g., d = 8 ), achieving superior performance and effective matching (Figure 5). 1067

4 See https://openreview.net/forum?id=Z8qk2iM5uLI . Weincludes the reproduced code and training logs of VQ-WAE in our supplementary materials.
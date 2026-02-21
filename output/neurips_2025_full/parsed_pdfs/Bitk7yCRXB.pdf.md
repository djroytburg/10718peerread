15

## Non-vacuous Bounds for the test error of Deep Learning without any change to the trained models

## Anonymous Author(s)

Affiliation Address email

## Abstract

Deep neural network (NN) with millions or billions of parameters can perform really well on unseen data, after being trained from a finite training set. Various prior theories have been developed to explain such excellent ability of NNs, but do not provide a meaningful bound on the test error. Some recent theories, based on PAC-Bayes and mutual information, are non-vacuous and hence promising to explain the excellent performance of NNs. However, they often require a stringent assumption and extensive modification (e.g. compression, quantization) to the trained model of interest. Therefore, those prior theories provide a guarantee for the modified versions only. In this paper, we propose two novel bounds on the test error of a model. Our bounds uses the training set only and require no modification to the model. Those bounds are verified on a large class of modern NNs, pretrained by Pytorch on the ImageNet dataset, and are non-vacuous. To the best of our knowledge, these are the first non-vacuous bounds at this large scale, without any modification to the pretrained models.

## 1 Introduction

Deep neural networks (NNs) are arguably the most effective families in Machine Learning. They have 16 been helping us to produce various breakthoughs, from mastering complex games [39], generating 17 high-quality languages [10] or images [20], protein structure prediction [22], to building multi-task 18 systems such as Gimini [41] and ChatGPT [1]. Big or huge NNs can efficiently learn knowledge 19 from large datasets and then perform extremely well on unseen data. 20

- Despite many empirical successes, there still remains a big gap between theory and practice of modern 21 NNs. In particular, it is largely unclear [48] about Why can deep NNs generalize well on unseen 22 data after being trained from a finite number of samples? This question relates to the generalization 23 ability of a trained model. The standard learning theories suffer from various difficulties to provide a 24 reasonable explanation. Various approaches have been studied, e.g. Radermacher complexity [18, 5], 25 algorithmic stability [38, 11], algorithmic robustness [47, 40], PAC-Bayes [32, 7]. 26
- Some recent theories [50, 7, 28-30] are really promissing, as they can provide meaningful bounds on 27

28

- the test error of some models. Dziugaite and Roy [14] obtained a non-vacuous bound by optimizing a
- distribution over NN parameters. [50, 16, 34] bounded the expected error of a stochastic NN by using 29
- off-the-shelf compression methods. Those theories follow the PAC-Bayes approach. On the other 30

31

hand, Nadjahi et al. [35] showed the potential of the stability-based approach. Although making a

- significant progress, those theories are meaningful for small and stochastic NNs only. 32
- Lotfi et al. [29, 30] made a significant step to analyze the generalization ability of big/huge NNs, 33
- such as large language models (LLM). Using state-of-the-art quantization, finetuning and some other 34

Table 1: Recent approaches for analyzing generalization error. glyph[check] means 'Required' or 'Yes'. The upper part shows the required assumptions about differrent aspects, e.g., hypothesis space, loss function, training or finetuning. The lower part reports non-vacuousness in different situations.

| Approach                                                 | Radermacher complexity [5]   | Alg. Stability [9, 27]   | Alg. Robustness [47, 23, 42]   | Mutual Info [46, 35]                   | PAC-Bayes [50, 34] [29,   | 30]                       | Ours                      |
|----------------------------------------------------------|------------------------------|--------------------------|--------------------------------|----------------------------------------|---------------------------|---------------------------|---------------------------|
| Requirement:                                             |                              |                          |                                |                                        |                           |                           |                           |
| Model compressibility Train or finetune                  | glyph[check]                 | glyph[check]             |                                | glyph[check] glyph[check] glyph[check] | glyph[check] glyph[check] | glyph[check] glyph[check] |                           |
| Lipschitz loss Finite hypothesis space                   |                              |                          |                                |                                        | glyph[check]              |                           |                           |
| Stochastic models only Trained models Training size > 1M |                              | glyph[check]             |                                | glyph[check]                           | glyph[check]              | glyph[check] glyph[check] | glyph[check] glyph[check] |

techniques, the PAC-Bayes bounds by [30, 29] are non-vacuous for huge LLMs, e.g., GPT-2 and 35 LLamMA2. Those bounds significantly push the frontier of deep learning theory. 36

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

In this work, we are interested in estimating or bounding the expected error F ( P, h ) of a specific model (hypothesis) h which is trained from a finite number of samples from distribution P . The expected error tells how well a model h can generalize on unseen data, and hence can explain the performance of a trained model. This estimation problem is fundamental in learning theory [33], but arguably challenging for NNs. Many prior theories [50, 28, 35] were developed for stochastic models , but not for a trained model h of interest. Lotfi et al. [29, 30] made a significant progress to remove 'stochasticity'. For example, Lotfi et al. [30] provided a non-vacuous bound for the 2-bit quantized (and finetuned) versions of LLamMA2. Nonetheless, those theories require to use a method for intensively quantizing or compressing h . This means that those theories are for the quantized or compressed models, and hence may not necessarily be true for the original (unquantized or uncompressed) models . This is a major limitation of those bounds. Such a limitation calls for novel theories that directly work with a given model h .

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

Our contributions in this work are as follow:

- We develop a novel bound on the expected error F ( P, h ) of a trained model h . This bound does not require stringent assumptions as prior bounds do. It encodes both the complexity of the data distribution and the behavior of model h at local areas of the data space.
- The main technical challenge to obtain our bound is to use the training set to approximate an intractable term which summarizes the true error of h at different local areas of the data space. We resolve this challenge by analyzing various properties of small and binomial random variables.
- We next derive a tractable bound that can be easily computed from the training set only, without any change to h . Hence this bound directly provides a guarantee for h . Those properties are really beneficial and enable our bound to overcome the major limitations of prior theories. Table 1 presents a more detailed comparison about some key aspects.
- Third, we develop a novel bound that uses a data transformation method. This bound can help us to analyze more properties of a trained model, and enable an effective comparison between two trained models. This bound may be useful in many contexts, where prior theories cannot provide an effective answer.
- Finally, we did an extensive evaluation for a large class of modern NNs which were pretrained by Pytorch on the ImageNet dataset with more than 1.2M images. The results show that our bounds are non-vacuous. To the best of our knowledge, this is the first time that a theoretical bound is non-vacuous at this large scale, without any change to the trained models.

Organization: The next section presents a comprehensive survey about related work, the main advan69 tages and limitations of prior theories. We then present our novel bounds in Section 3, accompanied 70 with more detailed comparisons. Section 4 contains our empirical evaluation for some pretrained NNs. 71 Section 5 concludes the paper. Proofs and more experimental details can be found in appendices. 72

- Notations: S often denotes a dataset and | S | denotes its size/cardinality. Γ denotes a partition of the 73

data space. [ K ] denotes the set { 1 , ..., K } of natural numbers at most K . glyph[lscript] denotes a loss function, 74 and h often denotes a model or hypothesis of interest. 75

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

## 2 Related work

Various approaches have been studied to analyze generalization capability, e.g., Radermacher complexity [4], algorithmic stability [38, 15], algorithmic robustness [47], Mutual-infomation based bounds [46, 35], PAC-Bayes [32, 19]. Those approaches connect different aspects of a learning algorithm or hypothesis (model) to generalization.

Norm-based bounds [5, 18, 17] is one of the earliest approaches to understand NNs. The existing studies often use Rademacher complexity to provide data- and model-dependent bounds on the generalization error. An NN with smaller weight norms will have a smaller bound, suggesting better generalization on unseen data. Nonetheless, the norms of weight matrices are often large for practical NNs [3]. Therefore, most existing norm-based bounds are vacuous.

Algorithmic stability [9, 38, 12, 24] is a crucial approach to studying a learning algorithm. Basically, those theories suggest that a more stable algorithm can generalize better. Stable algorithms are less likely to overfit the training set, leading to more reliable predictions. The stability requirement in those theories is that a replacement of one sample for the training set will not significantly change the loss of the trained model. Such an assumption is really strong. One primary drawback is that achieving stability often requires restricting model complexity, potentially sacrificing predictive accuracy on challenging datasets. Therefore, this approach has a limited success in understanding deep NNs.

Algorithmic robustness [47, 40, 23, 42] is a framework to study generalization capability. It essesntially says that a robust learning algorithm can produce robust models which can generalize well on unseen data. This approach provides another lens to understand a learning algorithm and a trained model. However, it requires the assumption that the learning algorithm is robust, i.e., the loss of the trained model changes little in the small areas around the training samples. Such an assumption is really strong and cannot apply well for modern NNs, since many practical NNs suffer from adversarial attacks [31, 49]. Than et al. [42] showed that those theories are often vacuous.

Neural Tangent Kernel [21] provides a theoretical lens to study generalization of NNs by linking them to kernel methods in the infinite-width limit. As networks grow wider, their training dynamics under gradient descent can be approximated by a kernel function which remains constant throughout training. This perspective simplifies the analysis of complex neural architectures. The framework enables explicit generalization bounds, and a deeper understanding of how network architecture and initialization affect learning. However, the main limitation of this framework comes from its assumptions, such as the infinite-width regime and fixed kernel during training, may not fully capture the behavior of finite, practical networks where feature learning is dynamic. Some other studies [25] can remove the infinite-width regime but assume the infinite depth .

Mutual information (MI) [46, 35] has emerged as a powerful tool for analyzing generalization by quantifying the dependency between a model's learned representations and the data. Since a trained model contains the (compressed) knowledge learned from the training samples, MI offers a principled framework for studying the trade-off between compression and predictive accuracy. However, the existing MI-based theories [46, 45, 37, 35] have a notable drawback: computing MI in high-dimensional, non-linear settings is computationally challenging. This drawback poses significant challenges for analyzing deep NNs, although [35] obtained some promissing results on small NNs.

PAC-Bayes [32, 19, 8] recently has received a great attention, and provide non-vacuous bounds [50, 34] for some NNs. Those bounds often estimate E ˆ h [ F ( P, ˆ h )] which is the expectation of the test error over the posterior distribution of ˆ h . It means that those bounds are for a stochastic model ˆ h . Hence they provide limited understanding for a specific deterministic model h . Neyshabur et al. [36] provided an attempt to derandomization for PAC-Bayes but resulted in vacuous bounds for modern neural networks [3]. Some recent attempts to derandomization include [44, 13].

Non-vacuous bounds for NNs: Dziugaite and Roy [14] obtained a non-vacuous bound for NNs by finding a posterior distribution over neural network parameters that minimizes the PAC-Bayes bound.

- Their optimized bound is non-vacuous for a stochastic MLP with 3 layers trained on MNIST dataset. 125

Zhou et al. [50] bounded the population loss of a stochastic NNs by using compressibility level of a 126

NN. Using off-the-shelf neural network compression schemes, they provided the first non-vacuous 127

bound for LeNet-5 and MobileNet, trained on ImageNet with more than 1.2M samples. Lotfi et al. 128

[28] developed a compression method to further optimize the PAC-Bayes bound, and estimated 129

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

the error rate of 40.9% for MobileViT on ImageNet.

Mustafa et al. [34] provided a non-vacuous

PAC-Bayes bound for adversarial population loss for VGG on CIFAR10 dataset. Galanti et al. [16]

presented a PAC-Bayes bound which is non-vacuous for Convolutional NNs with up to 20 layers and for CIFAR10 and MNIST. Akinwande et al. [2] provided a non-vacuous PAC-Bayes bound

for prompts. Although making a significant progress for NNs, those bounds are non-vacuous for stochastic neural networks only. Biggs and Guedj [7] provided PAC-Bayes bounds for deterministic

models and obtain (empirically) non-vacuous bounds for a specific class of (SHEL) NNs with a single hidden layer, trained on MNIST and Fashion-MNIST. Nonetheless, it is unclear about how well those

bounds apply to bigger or deeper NNs.

Towards understanding big/huge NNs, Lotfi et al. [29, 30] made a significant step that provides non-vacuous bounds for LLMs. While the PAC-Bayes bound in [29] can work with LLMs trained from i.i.d data, the recent bound in [30] considers token-level loss for LLMs and applies to dependent settings, which is close to the practice of training LLMs. Using both model quantization, finetuning and some other techniques, the PAC-Bayes bound by [30] is shown to be non-vacuous for huge LLMs, e.g., LLamMA2. Those bounds significantly push the frontier of learning theory towards building a solid foundation for DL.

Nonetheless, there are two main drawbacks of those bounds [29, 30]. First, model quantization or compression is required in order to obtain a good bound. It means, those bounds are for the quantized or compressed models, and hence may not necessarily be true for the original (unquantized or uncompressed) models . For example, [30] provided a non-vacuous bound for the 2-bit quantized versions of LLamMA2, instead of their original pretrained versions. Second, those bounds require the assumption that the model (hypothesis) family is finite , meaning that a learning algorithm only searches in a space with finite number of specific models. Although such an assumption is reasonable for the current computer architectures, those bounds cannot explain a trained model that belongs to families with infinite (or uncountable) number of members, which are provably prevalent. In contrast, our bounds apply directly to any specific model without requiring any modification or support. A comparison between our bounds and prior approaches about some key aspects is presented in Table 1.

## 3 Error bounds

In this section, we present three novel bounds for the expected error of a given model. The first bound provides a general form which directly depends on the complexity of the data distribution and the trained model. The second bound provides an explicit upper bound for the error, which can be computed directly from any given dataset. The last bound helps us to analyze the robustness of a model by using data augmentation.

Consider a hypothesis (or model) h , defined on an instance set Z , and a nonnegative loss function glyph[lscript] . Each glyph[lscript] ( h , z ) tells the loss (or quality) of h at an instance z ∈ Z . Given a distribution P defined on Z , the quality of h is measured by its expected loss F ( P, h ) = E z ∼ P [ glyph[lscript] ( h , z )] . Quantity F ( P, h ) tells the generalization ability of model h ; a smaller F ( P, h ) implies that h can generalize better on unseen data.

For analyzing generalization ability, we are often interested in estimating (or bounding) F ( P, h ) . Sometimes this expected loss is compared with the empirical loss of h on a data set S = { z 1 , ..., z n } ⊆ Z , which is defined as F ( S , h ) = 1 n ∑ z ∈ S glyph[lscript] ( h , z ) . Note that a small F ( S , h ) does not neccessarily imply good generalization of h , since overfitting may appear. Therefore, our ultimate goal is to estimate F ( P, h ) directly.

Let Γ( Z ) := ⋃ K i =1 Z i be a partition of Z into K disjoint nonempty subsets. Denote S i = S ∩ Z i , and n i = | S i | as the number of samples falling into Z i , meaning that n = ∑ K j =1 n j . Denote T = { i ∈ [ K ] : n i &gt; 0 } , a i ( h ) = E z [ glyph[lscript] ( h , z ) | z ∈ Z i ] for i ∈ [ K ] , and a o = max j / ∈ T a j ( h ) .

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

## 3.1 General bound

The first result incorporates the properties of the data distribution and the trained model.

Theorem 3.1. Given a partition Γ and a bounded nonnegative loss glyph[lscript] , consider a model h which may depend on a dataset S with n i.i.d. samples from distribution P . Denote p i = P ( Z i ) as the measure of area Z i for i ∈ [ K ] , and u = ∑ K i =1 γnp i (1 + γnp i ) . For any constants γ ≥ 1 , δ 1 ≥ exp( -u ln γ 4 n -3 ) and δ 2 &gt; 0 , we have the following with probability at least 1 -δ 1 -δ 2 :

<!-- formula-not-decoded -->

where g (Γ , h , δ 2 ) = √ ln(2 K/δ 2 ) n ∑ i ∈ T √ n i ( a o + √ 2 a i ( h ) ) + 2 ln(2 K/δ 2 ) n ( a o | T | + ∑ i ∈ T a i ( h )) and C = sup z ∈Z glyph[lscript] ( h , z ) .

This theorem suggests that the expected loss cannot be far from the empirical loss F ( S , h ) . The gap between the two is at most C √ u 2 n 2 ln 1 δ 1 + g (Γ , h , δ 2 ) . Such a gap represents the uncertainty of our bound and mostly depends on the sample size n , the trained model h , the data distribution P and the partition Γ . We emphasize that bound (1) has some interesting properties:

- First, it does not require any assumption about the hypothesis family and learning algorithm. This is an advantage over many approaches including algorithmic stability [9, 27], robustness [47, 23], Radermacher complexity [4, 5]. This bound focuses directly on the the model h of interest, helping it to be tighter than many prior bounds.
- Second, it depends on the complexity of the data distribution. Note that u encodes the complexity of P . For a uniform partition Γ , a more structured distribution P can have a higher sum ∑ K i =1 p 2 i . As an example of structured distributions, a Gaussian with a small variance has the most probability density in a small area around its mean and lead to a high p i for some i . Meanwhile a less structured distribution (e.g. uniform) can produce a small ∑ K i =1 p 2 i and hence smaller u . To the best of our knowledge, such an explicit dependence on the distribution complexity is rare in prior theories.
- Third, it is model-dependent. Some particular properties of model h are encoded in g (Γ , h , δ 2 ) and the empirical loss . A better model h will lead to smaller a i 's and hence g . On the other hand, a worse model can have a bigger g , leading to a higher RHS of (1).

It is worth noticing the similarity between our bound (1) and robustness-based bounds in [23, 42]. F ( S , h ) + g (Γ , h , δ 2 ) is the common part in those bounds. Our bound (1) contains C √ u 2 n 2 ln 1 δ 1 that encodes the complexity of the data distribution, whereas the bounds in [23, 42] use a robustness quantity that measures the sensitivity of the loss w.r.t. a change in the input. While prior bounds are not amenable to be exactly computed from a training set, our bound enables to easily derive a computable and non-vacuous bound (below). This is the main advantage of bound (1).

Proof sketch. The detailed proof can be found in Appendix A. We focus on bounding the probability Pr( F ( P, h ) -F ( S , h ) ≥ φ ) , for some gap φ . Note that F ( P, h ) -F ( S , h ) = A + B , where A = F ( P, h ) -∑ i n i n a i ( h ) and B = ∑ i n i n a i ( h ) -F ( S , h ) . Therefore, our proof estimates Pr( A ≥ g ) and

<!-- formula-not-decoded -->

for some constant t . Once they are known, we can use the union bound to obtain a bound on Pr( F ( P, h ) -F ( S , h ) ≥ g + t ) as desired. We use a result from [23] to bound Pr( A ≥ g ) . The remaining task is to estimate (2), which is the main challenge . This challenge requires approximating an intractable quantity from a data set.

We resolve this challenge by developing Theorem A.1. Its proof contains three main steps:

1. First we show Pr( B ≥ t ) ≤ e -yt E h , n [ E S [ e yB | h , n ]] , for n = { n 1 , ..., n K } and some y .

2. We next estimate E S [ e yB K | h , n ] . Overall, we make sure that E S [ e yB | h , n ] ≤ e ψ ( y, n ) , for some function ψ ( y, v ) which does not depend on h . As a result Pr( B ≥ t ) ≤ E v e ψ ( y, n ) .

3. The last step is to bound E n e ψ ( y, n ) . This requires us to develop various analyses for small random variables in Appendix B. A suitable choice for t, y completes our proof.

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

## 3.2 Tractable bounds

It is worth noticing that bound (1) contains some unknown quantities, e.g., u and a i 's, which cannot be computed exactly. This is its main limitation. The following bound overcomes such a limitation.

Theorem 3.2. Given the notations and assumption in Theorem 3.1, for any constants γ ≥ 1 , δ &gt; 0 and α ∈ [0 , γn ( K + γn ) K (4 n -3) ] , we have the following with probability at least 1 -γ -α -δ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

One special property is that we can evaluate our bound easily by using only the training set . Indeed, we can choose K and a specific partition Γ of the data space. Then we can count n i and T and evaluate the bound (3) easily. This property is remarkable and beneficial in practice.

A theoretical comparison with closely related bounds: Although many model-dependent bounds [23, 42, 7, 44, 29, 30] have been proposed, our bound (3) has various advantages:

- Mild assumption: Our bound does not require stringent assumptions as in prior ones. Some prior bounds require stability [27, 26] or robustness [47, 23, 40] of the learning algorithm. Those assumptions are often violated in practice, e.g. for the appearance of adversarial attacks [49]. Some theories [29, 30] assume that the hypothesis class is finite, which is restrictive. In contrast, our bound requires only i.i.d. assumption which also appears in most prior bounds.
- Easy evaluation: An evaluation of our bound (3) will be simple and does not require any modification to the model h of interest. This is a crucial advantage. Many prior theories require intermediate steps to change the model of interest into a suitable form. For example, state-of-the-art methods to compress NNs are required for [50, 28, 35]; quantization for a model is required for [29, 30]; finetuning (e.g. SubLoRA) is required for [29, 30]. Those facts suggests that evaluations for prior bounds are often expensive. Besides, many prior model-dependent bounds [47, 23, 42] cannot be exactly computed from a training set only.
- No change to the model: Most prior non-vacuous bounds [50, 14, 29, 30] require extensively compressing (or quantizing) model h of interest and then retraining/finetuning the compressed version. Sometimes the compression step is too restrictive and produces low-quality models [29]. Therefore, a modification will change model h and hence those bounds do not directly provide guarantees for the generalization ability of h . In contrast, our bound (3) does not require any change to model h , and hence directly provides a guarantee for h . √

There is a nonlinear relationship between K and the uncertainty term Unc (Γ) = C ˆ uα ln γ + g 2 ( δ/ 2) in our bound. A partition with a larger K can make the sum ∑ K i =1 ( n i n ) 2 smaller, as the samples can be spread into more areas. However a larger K can make g 2 ( δ ) larger. Therefore, we should not choose too large K . On the other hand, a small K can make the sum ∑ K i =1 ( n i n ) 2 large, since more samples can appear in each area Z i and enlarge n i n . Therefore, we should not choose too small K . Furthermore, we need to choose constant α carefully, since there is a trade-off in the bound and the certainty 1 -γ -α -δ . A smaller α can make the bound smaller, but could enlarge γ -α and hence reduce the certainty of the bound.

The next result considers the robustness of a model.

Theorem 3.3. Given the assumption in Theorem 3.2, let ˆ S = T ( S ) be the result of using a transformation method T , which is independent with h , on the samples of S . Denote ¯ glyph[epsilon1] ( h ) = ∑ i ∈ T m i m ¯ glyph[epsilon1] i

<!-- formula-not-decoded -->

for each i ∈ T . We have the following with probability at least 1 -γ -α -δ :

<!-- formula-not-decoded -->

This theorem suggests that a model can be better if its loss is less sensitive with respect to some small 265 changes in the training samples . This can be seen from each quantity ¯ glyph[epsilon1] i which measures the average 266

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

ˆ

difference of the loss of h for the samples S i and S i belonging to the same small area. This result closely relates to adversarial training [31], where one often wants to train a model which is robust w.r.t small changes in the inputs. It is also worth noticing that if T transforms S too much, both the loss F ( ˆ S , h ) and the sensitivity ¯ glyph[epsilon1] can be large. As a result, the bound (4) will be large. In fact, our proof suggests that bound (4) is worse than bound (3).

The main benefit of Theorem 3.3 is that we can use some transformation methods to compare some trained models. This is particularly useful for the cases where two models have comparable (even zero) training losses. For those cases, Theorem 3.2 does not provide a satisfactory answer. Instead, we can use a simple augmentation method (e.g., noise perturbation, rotation, translation, ...) to produce a dataset ˆ S and then use this dataset to evaluate the upper bound (4). By this way, we use both the training loss F ( S , h ) and ¯ glyph[epsilon1] ( h ) + F ( ˆ S , h ) + ∑ i ∈ T ( n i n -m i m ) F ( S i , h ) for comparison.

## 4 Empirical evaluation

In this section, we present two sets of extensice evaluations about the our bounds. We use 32 modern NN models 1 which were pretrained by Pytorch on the ImageNet dataset with 1,281,167 images. Those models are multiclass classifiers. Our main aim is to provide a guarantee for the error of a trained model, without any further modification. Therefore, no prior bound is taken into comparison, since those existing bounds are either already vacuous or require some extensive modifications or cannot directly apply to those trained NNs.

## 4.1 Large-scale evaluation for pretrained models

The first set of experiments verifies nonvacuouness of our first bound (3) and the effects of some parameters in the bound. We use the training part of ImageNet only to compute the bound.

Experimental settings: We fix δ = 0 . 01 , α = 100 , γ = 0 . 04 -1 /α . This choice means that our bound is correct with probability at least 95%. The partition Γ is chosen with K = 200 small areas of the input space, by clustering the training images into 200 areas, whose centroids are initialized randomly. The upper bound (3) for each model was computed with 5 random seeds. We use the 0-1 loss function, meaning that our bound directly estimates the true classification error.

Results: The overall results are reported in Table 2. One can observe that our bound for all models are all non-vacuous even for the non-optimized choices of some parameters. Our estimate is often 2-3 times higher than the oracle test error of each model. When choosing the best parameter for each model by grid search, we can obtain much better bounds about the test errors. Note that non-vacuousness of our bound holds true for a large class of deep NN families, some of which have more than 630M parameters. To the best of our knowledge, bound (3) is the first theoretical bound which is non-vacuous at such a large scale, without requiring any modification to the trained models.

Effect of parameters: Note that our bound depends on the choice of some parameters. Figure 1 reports the changes of ∑ K i =1 ( n i n ) 2 as the partition Γ changes. We can see that this quantity tends to decrease as we divide the input space into more small areas. Meanwhile, Figure 2 reports the uncertainty term, as either α or K changes. Observe that a larger K can increase the uncertainty fast, while an increase in α can gradually decrease the uncertainty. Those figures enable an easy choice for the parameters in our bound.

## 4.2 Evaluation with data augmentation

As mentioned before, our bound (3) can provide a theoretical certificate for a trained model, but may not be ideal to compare two models which have the same training error. Sometimes, a model can have a lower training error but a higher test error (such as DenseNet161 vs. DenseNet201, VIT L 16 linear vs. VIT L 16 V1). Bound (3) may not be good for model comparison. In those cases, we need to use bound (4) for comparison.

Experimental settings: We fix δ = 0 . 01 , α = 100 , γ = 0 . 04 -1 /α , K = 200 as before. We use white noise addition as the transformation method in Theorem 3.3. Specifically, each image is added

1 https://pytorch.org/vision/stable/models.html

314

315

316

317

318

319

Table 2: Upper bounds on the true error (in %) of 32 deep NNs which were pretrained on ImageNet dataset. The second column presents the model size, the third column contains the test accuracy at Top 1, as reported by Pytorch. 'Mild" reports the bound for the choice of { δ = 0 . 01 , K = 200 , α = 100 , γ = 0 . 04 -1 /α }, while 'Optimized" reports the bound with parameter optimization by grid search. The grid search is done for K ∈ { 100 , 200 , 300 , 400 , 500 , 1000 , 5000 , 10000 } , α ∈ { 10 , 20 , ..., 100 } , δ = 0 . 01 and γ = 0 . 04 -1 /α . The last two columns report our estimates about the true error, with a certainty at least 95%.

| Model                 | #Params (M)   | Training error   | Acc@1   | Test error   | Error bound (3)   | Error bound (3)   |
|-----------------------|---------------|------------------|---------|--------------|-------------------|-------------------|
|                       |               |                  |         |              | Mild              | Optimized         |
| ResNet18 V1           | 11.7          | 21.245           | 69.758  | 30.242       | 57.896 ± 4.189    | 54.262            |
| ResNet34 V1           | 21.8          | 15.669           | 73.314  | 26.686       | 52.320 ± 4.189    | 48.686            |
| ResNet50 V1           | 25.6          | 13.121           | 76.130  | 23.870       | 49.772 ± 4.189    | 46.138            |
| ResNet101 V1          | 44.5          | 10.502           | 77.374  | 22.626       | 47.153 ± 4.189    | 43.519            |
| ResNet152 V1          | 60.2          | 10.133           | 78.312  | 21.688       | 46.784 ± 4.189    | 43.150            |
| ResNet50 V2           | 25.6          | 8.936            | 80.858  | 19.142       | 45.587 ± 4.189    | 41.953            |
| ResNet101 V2          | 44.5          | 6.008            | 81.886  | 18.114       | 42.659 ± 4.189    | 39.025            |
| ResNet152 V2          | 60.2          | 5.178            | 82.284  | 17.716       | 41.829 ± 4.189    | 38.195            |
| SwinTransformer B     | 87.8          | 6.464            | 83.582  | 16.418       | 43.115 ± 4.189    | 39.481            |
| SwinTransformer B V2  | 87.9          | 6.392            | 84.112  | 15.888       | 43.043 ± 4.189    | 39.409            |
| SwinTransformer T     | 28.3          | 9.992            | 81.474  | 18.526       | 46.643 ± 4.189    | 43.009            |
| SwinTransformer T V2  | 28.4          | 8.724            | 82.072  | 17.928       | 45.375 ± 4.189    | 41.741            |
| VGG13                 | 133.0         | 18.456           | 69.928  | 30.072       | 55.107 ± 4.189    | 51.473            |
| VGG13 BN              | 133.1         | 19.223           | 71.586  | 28.414       | 55.874 ± 4.189    | 52.240            |
| VGG19                 | 143.7         | 16.121           | 72.376  | 27.624       | 52.772 ± 4.189    | 49.138            |
| VGG19 BN              | 143.7         | 15.941           | 74.218  | 25.782       | 52.592 ± 4.189    | 48.958            |
| DenseNet121           | 8.0           | 15.631           | 74.434  | 25.566       | 52.282 ± 4.189    | 48.648            |
| DenseNet161           | 28.7          | 10.48            | 77.138  | 22.862       | 47.131 ± 4.189    | 43.497            |
| DenseNet169           | 14.1          | 12.395           | 75.600  | 24.400       | 49.046 ± 4.189    | 45.412            |
| DenseNet201           | 20.0          | 9.806            | 76.896  | 23.104       | 46.457 ± 4.189    | 42.823            |
| ConvNext Base         | 88.6          | 5.209            | 84.062  | 15.938       | 41.860 ± 4.189    | 38.226            |
| ConvNext Large        | 197.8         | 3.846            | 84.414  | 15.586       | 40.497 ± 4.189    | 36.863            |
| RegNet Y 128GF e2e    | 644.8         | 5.565            | 88.228  | 11.772       | 42.216 ± 4.189    | 38.582            |
| RegNet Y 128GF linear | 644.8         | 9.032            | 86.068  | 13.932       | 45.683 ± 4.189    | 42.049            |
| RegNet Y 32GF e2e     | 145.0         | 7.127            | 86.838  | 13.162       | 43.778 ± 4.189    | 40.144            |
| RegNet Y 32GF linear  | 145.0         | 10.558           | 84.622  | 15.378       | 47.209 ± 4.189    | 43.575            |
| RegNet Y 32GF V2      | 145.0         | 3.761            | 81.982  | 18.018       | 40.412 ± 4.189    | 36.778            |
| VIT B 16 linear       | 86.6          | 14.969           | 81.886  | 18.114       | 51.620 ± 4.189    | 47.986            |
| VIT B 16 V1           | 86.6          | 5.916            | 81.072  | 18.928       | 42.567 ± 4.189    | 38.933            |
| VIT H 14 linear       | 632.0         | 9.951            | 85.708  | 14.292       | 46.602 ± 4.189    | 42.968            |
| VIT L 16 linear       | 304.3         | 11.003           | 85.146  | 14.854       | 47.654 ± 4.189    | 44.020            |
| VIT L 16 V1           | 304.3         | 3.465            | 79.662  | 20.338       | 40.116 ± 4.189    | 36.482            |

<!-- image -->

Figure 1: The dynamic of ˆ n = ∑ K i =1 ( n i n ) 2 as K changes.

<!-- image -->

K

Figure 2: The uncertainty Unc (Γ) = C √ ˆ uα ln γ + g ( δ/ 2) as (right) K changes and (left) α changes, for fixed K = 200 , γ = 0 . 04 -1 /α , δ = 0 . 01 .

by a noise which is randomly sampled from the normal distribution with mean 0 and variance σ 2 . Those noisy images are used to compute bound (4).

Results: Table 3 reports bound (4) for σ = 0 . 15 , ignoring the uncertainty part which is common for all models. One can observe that our bound (4) correlates very well with the test error of each model, except RegNet and VIT families. This suggests that the use of data augmentation can help us to better compare the performance of two models.

We next vary σ ∈ { 0 , 0 . 05 , 0 . 1 , 0 . 15 , 0 . 2 } to see when the noise can enable a good comparison. 320 Figure 3 reports the results about two families. We observe that while DenseNet161 has higher 321 training error than DenseNet201 does, the error bound for DenseNet161 tends to be lower than that 322

Table 3: Bound (4) on the test error (in %) of some models which were pretrained on ImageNet dataset. Each bound was computed by adding Gaussian noises to the training images, with σ = 0 . 15 .

| Model                |   Training error |   Test error |   Bound (4) |
|----------------------|------------------|--------------|-------------|
| ResNet18 V1          |           21.245 |       30.242 |     129.226 |
| ResNet34 V1          |           15.669 |       26.686 |     111.521 |
| DenseNet161          |           10.48  |       22.862 |      94.045 |
| DenseNet169          |           12.395 |       24.4   |     100.747 |
| DenseNet201          |            9.806 |       23.104 |      96.221 |
| VGG 13               |           18.456 |       30.072 |     142.87  |
| VGG 13 BN            |           19.223 |       28.414 |     134.955 |
| RegNet Y 32GF e2e    |            7.127 |       13.162 |      72.474 |
| RegNet Y 32GF linear |           10.558 |       15.378 |      85.368 |
| RegNet Y 32GF V2     |            3.761 |       18.018 |      67.764 |
| VIT B 16 linear      |           14.969 |       18.11  |      96.967 |
| VIT B 16 V1          |            5.916 |       18.93  |      65.969 |
| VIT L 16 linear      |           11.003 |       14.85  |      80.178 |
| VIT L 16 V1          |            3.465 |       20.34  |      58.402 |

Figure 3: The dynamic of bound (4) as the noise level σ increases. These subfigures report the main part ¯ glyph[epsilon1] ( h ) + F ( ˆ S , h ) of the bound.

<!-- image -->

of DenseNet201 as the images get more noisy. This suggests that DenseNet161 should be better than 323 DenseNet201, which is correctly reflected by their test errors. The same behavior also appears for 324 VGG13 and VGG13 BN. However, those two families require two different values of σ (0.05 for 325 VGG; 0.1 for DenseNet) to exhibit an accurate comparison. This also suggests that the anti-correlation 326 mentioned before for RegNet and VIT may be due to the small value of σ in Table 3. Those two 327 families may require a higher σ to exhibit an accurate comparison. 328

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

## 5 Conclusion

Providing theoretical guarantees for the performance of a model in practice is crucial to build reliable ML applications. Our work contributes three bounds on the test error of a model, one of which is non-vacuous for all the trained deep NNs in our experiments, without requiring any change to the trained models. Hence, our bounds can be used to provide a non-vacuous theoretical certificate for a trained model. This fills in the decade-missing cornerstone of deep learning theory.

Our work opens various avenues for future research. Indeed, while the the uncertainty part of bound (1) depends on the inherent property of the model of interest, that in bound (3) mostly does not. This suggests that bound (3) is suboptimal. One direction to develop better theories is to take more properties of a model into consideration, e.g. exploit more fine-grained properties of bound (1). Another direction is to take dependency of the training samples into account. However, it may require some improvements from very fundamental steps, e.g., concentrations for dependent variables. Since our bounds are for general settings, one interesting direction is to provide certificates for models in different types of applications, e.g. regression, segmentation, language inference, translation, text-2-images, image-2-text, ... We believe that our bounds provide a good starting point for those directions.

345

346

## References

- [1] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida,
- J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint 347 arXiv:2303.08774 , 2023. 348

349

350

351

- [2] V. Akinwande, Y. Jiang, D. Sam, and J. Z. Kolter. Understanding prompt engineering may not require rethinking generalization. In International Conference on Learning Representations , 2024.
- [3] S. Arora, R. Ge, B. Neyshabur, and Y. Zhang. Stronger generalization bounds for deep nets via 352 a compression approach. In International Conference on Machine Learning , pages 254-263. 353 PMLR, 2018. 354
- [4] P. L. Bartlett and S. Mendelson. Rademacher and gaussian complexities: Risk bounds and 355 structural results. Journal of Machine Learning Research , 3(Nov):463-482, 2002. 356
- [5] P. L. Bartlett, D. J. Foster, and M. J. Telgarsky. Spectrally-normalized margin bounds for neural 357 networks. Advances in Neural Information Processing Systems , 30:6240-6249, 2017. 358
- [6] M. Belkin, D. Hsu, S. Ma, and S. Mandal. Reconciling modern machine-learning practice and 359 the classical bias-variance trade-off. Proceedings of the National Academy of Sciences , 116 360 (32):15849-15854, 2019. 361
- [7] F. Biggs and B. Guedj. Non-vacuous generalisation bounds for shallow neural networks. In 362 International Conference on Machine Learning , pages 1963-1981. PMLR, 2022. 363
- [8] F. Biggs and B. Guedj. Tighter pac-bayes generalisation bounds by leveraging example difficulty. 364 In International Conference on Artificial Intelligence and Statistics , pages 8165-8182. PMLR, 365 2023. 366

367

368

- [9] O. Bousquet and A. Elisseeff. Stability and generalization. The Journal of Machine Learning Research , 2:499-526, 2002.

369

370

371

372

373

374

375

376

- [10] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems , volume 33, pages 1877-1901, 2020.
- [11] A. Brutzkus and A. Globerson. An optimization and generalization analysis for max-pooling networks. In Uncertainty in Artificial Intelligence , pages 1650-1660. PMLR, 2021.
- [12] Z. Charles and D. Papailiopoulos. Stability and generalization of learning algorithms that converge to global optima. In International Conference on Machine Learning , pages 745-754. PMLR, 2018.
- [13] E. Clerico, T. Farghly, G. Deligiannidis, B. Guedj, and A. Doucet. Generalisation under gradient 377 descent via deterministic pac-bayes. In International Conference on Algorithmic Learning 378 Theory , 2025. 379

380

381

382

- [14] G. K. Dziugaite and D. M. Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. In Conference on Uncertainty in Artificial Intelligence (UAI) , 2017.
- [15] V. Feldman and J. Vondrak. High probability generalization bounds for uniformly stable 383 algorithms with nearly optimal rate. In Conference on Learning Theory (COLT) , pages 1270384 1279. PMLR, 2019. 385
- [16] T. Galanti, L. Galanti, and I. Ben-Shaul. Comparative generalization bounds for deep neural 386 networks. Transactions on Machine Learning Research , 2023. 387
- [17] T. Galanti, M. Xu, L. Galanti, and T. Poggio. Norm-based generalization bounds for sparse 388 neural networks. Advances in Neural Information Processing Systems , 36, 2023. 389
- [18] N. Golowich, A. Rakhlin, and O. Shamir. Size-independent sample complexity of neural 390 networks. Information and Inference: A Journal of the IMA , 9(2):473-504, 2020. 391

- [19] M. Haddouche and B. Guedj. Pac-bayes generalisation bounds for heavy-tailed losses through 392 supermartingales. Transactions on Machine Learning Research , 2023. 393
- [20] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In Advances in Neural 394 Information Processing Systems , volume 33, pages 6840-6851, 2020. 395

396

397

398

- [21] A. Jacot, F. Gabriel, and C. Hongler. Neural tangent kernel: convergence and generalization in neural networks. In Advances in Neural Information Processing Systems , pages 8580-8589, 2018.

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

- [22] J. Jumper, R. Evans, A. Pritzel, T. Green, M. Figurnov, O. Ronneberger, K. Tunyasuvunakool, R. Bates, A. Žídek, A. Potapenko, et al. Highly accurate protein structure prediction with alphafold. Nature , 596(7873):583-589, 2021.
- [23] K. Kawaguchi, Z. Deng, K. Luh, and J. Huang. Robustness implies generalization via datadependent generalization bounds. In International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 10866-10894. PMLR, 2022.
- [24] I. Kuzborskij and C. Lampert. Data-dependent stability of stochastic gradient descent. In International Conference on Machine Learning , pages 2815-2824. PMLR, 2018.
- [25] J. Lee, J. Y . Choi, E. K. Ryu, and A. No. Neural tangent kernel analysis of deep narrow neural networks. In International Conference on Machine Learning , pages 12282-12351, 2022.
- [26] Y. Lei and Y. Ying. Fine-grained analysis of stability and generalization for stochastic gradient descent. In International Conference on Machine Learning , pages 5809-5819. PMLR, 2020.
- [27] S. Li, B. Zhu, and Y. Liu. Algorithmic stability unleashed: Generalization bounds with unbounded losses. In International Conference on Machine Learning , 2024.
- [28] S. Lotfi, M. Finzi, S. Kapoor, A. Potapczynski, M. Goldblum, and A. G. Wilson. Pac-bayes compression bounds so tight that they can explain generalization. In Advances in Neural Information Processing Systems , volume 35, pages 31459-31473, 2022.
- [29] S. Lotfi, M. A. Finzi, Y. Kuang, T. G. Rudner, M. Goldblum, and A. G. Wilson. Non-vacuous generalization bounds for large language models. In International Conference on Machine Learning , 2024.
- [30] S. Lotfi, Y . Kuang, M. A. Finzi, B. Amos, M. Goldblum, and A. G. Wilson. Unlocking tokens as data points for generalization bounds on larger language models. In Advances in Neural Information Processing Systems , 2024.
- [31] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations , 2018.
- [32] D. A. McAllester. Some pac-bayesian theorems. Machine Learning , 37(3):355-363, 1999.
- [33] M. Mohri, A. Rostamizadeh, and A. Talwalkar. Foundations of Machine Learning . MIT Press, 2018.
- [34] W. Mustafa, P. Liznerski, A. Ledent, D. Wagner, P. Wang, and M. Kloft. Non-vacuous generalization bounds for adversarial risk in stochastic neural networks. In International Conference on Artificial Intelligence and Statistics , pages 4528-4536, 2024.
- [35] K. Nadjahi, K. Greenewald, R. B. Gabrielsson, and J. Solomon. Slicing mutual information generalization bounds for neural networks. In International Conference on Machine Learning , 2024.
- [36] B. Neyshabur, S. Bhojanapalli, and N. Srebro. A pac-bayesian approach to spectrally-normalized margin bounds for neural networks. In International Conference on Learning Representations , 2018.
- [37] M. Sefidgaran, A. Gohari, G. Richard, and U. Simsekli. Rate-distortion theoretic generalization bounds for stochastic learning algorithms. In Conference on Learning Theory , pages 4416-4463. PMLR, 2022.

- [38] S. Shalev-Shwartz, O. Shamir, N. Srebro, and K. Sridharan. Learnability, stability and uniform 439 convergence. The Journal of Machine Learning Research , 11:2635-2670, 2010. 440
- [39] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, 441 I. Antonoglou, V. Panneershelvam, M. Lanctot, et al. Mastering the game of go with deep neural 442 networks and tree search. Nature , 529(7587):484-489, 2016. 443

444

445

- [40] J. Sokoli´ c, R. Giryes, G. Sapiro, and M. R. Rodrigues. Robust large margin deep neural networks. IEEE Transactions on Signal Processing , 65(16):4265-4280, 2017.

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

- [41] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 , 2023.
- [42] K. Than, D. Phan, and G. Vu. Gentle local robustness implies generalization. Machine Learning , 114(6):142, 2025.
- [43] A. Tseng, J. Chee, Q. Sun, V. Kuleshov, and C. De Sa. Quip#: Even better llm quantization with hadamard incoherence and lattice codebooks. In International Conference on Machine Learning , 2024.
- [44] P. Viallard, P. Germain, A. Habrard, and E. Morvant. A general framework for the practical disintegration of pac-bayesian bounds. Machine Learning , 113(2):519-604, 2024.
- [45] B. Wang, H. Zhang, J. Zhang, Q. Meng, W. Chen, and T.-Y. Liu. Optimizing informationtheoretical generalization bound via anisotropic noise of sgld. In Advances in Neural Information Processing Systems , volume 34, pages 26080-26090, 2021.
- [46] A. Xu and M. Raginsky. Information-theoretic analysis of generalization capability of learning algorithms. In Advances in Neural Information Processing Systems , volume 30, 2017.
- [47] H. Xu and S. Mannor. Robustness and generalization. Machine Learning , 86(3):391-423, 2012.
- [48] C. Zhang, S. Bengio, M. Hardt, B. Recht, and O. Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM , 64(3):107-115, 2021.
- [49] S. Zhou, C. Liu, D. Ye, T. Zhu, W. Zhou, and P. S. Yu. Adversarial attacks and defenses in deep learning: From a perspective of cybersecurity. ACM Computing Surveys , 55(8):1-39, 2022.
- [50] W. Zhou, V. Veitch, M. Austern, R. P. Adams, and P. Orbanz. Non-vacuous generalization bounds at the imagenet scale: a pac-bayesian compression approach. In International Conference on Learning Representations (ICLR) , 2019.

## A Proofs for main results 469

Proof of Theorem 3.1. We first observe that 470

<!-- formula-not-decoded -->

Next, we consider F ( P, h ) -∑ K i =1 n i n a i ( h ) = ∑ K i =1 p i a i ( h ) -∑ K i =1 n i n a i ( h ) = 471 ∑ K i =1 a i ( h ) [ p i -n i n ] . Note that ( n 1 , ..., n K ) is a multinomial random variable with pa472 rameters n and ( p 1 , ..., p K ) . Therefore, according to Lemma 7 in [23], we have 473 Pr ( ∑ K i =1 a i ( h ) [ p i -n i n ] &gt; g (Γ , h , δ 2 ) ) &lt; δ 2 . This implies 474

<!-- formula-not-decoded -->

On the other hand, Theorem A.1 below shows that 475

<!-- formula-not-decoded -->

Combining this with (6) and the union bound, we have 476

<!-- formula-not-decoded -->

477

completing the proof.

Proof of Theorem 3.2. Theorem 3.1 shows that 478

<!-- formula-not-decoded -->

479

480

481

482

483

where u and δ 1 depend on the sum ∑ K i =1 p 2 i . We next bound this quantity using S .

Since p i ≥ 0 and ∑ K i =1 p i = 1 , we can use the Lagrange multiplier method to show that ∑ K i =1 p 2 i is minimized at 1 /K . Hence u = ∑ K i =1 γnp i (1+ γnp i ) = γn + γ 2 n 2 ∑ K i =1 p 2 i ≥ γn + γ 2 n 2 /K . This suggests that exp( -u ln γ 4 n -3 ) ≤ exp( -( γn + γ 2 n 2 /K ) ln γ 4 n -3 ) ≤ exp( -γn ( K + γn ) ln γ K (4 n -3) ) ≤ γ -α . Choosing δ 1 = γ -α and plugging it into (9) lead to

<!-- formula-not-decoded -->

It is easy to see that g (Γ , h , δ/ 2) ≤ g 2 ( δ/ 2) , since a o ( h ) ≤ C and a i ( h ) ≤ C for any i . Therefore 484

<!-- formula-not-decoded -->

Next we consider u 2 n 2 = γ 2 n + γ 2 2 ∑ K i =1 p 2 i . Since S contains n i.i.d. samples, ( n 1 , ..., n K ) is a 485 multinomial random variable with parameters n and ( p 1 , ..., p K ) . Lemma B.8 shows 486

<!-- formula-not-decoded -->

Therefore Pr ( u 2 n 2 &gt; γ 2 n + γ 2 2 ∑ K i =1 ( n i n ) 2 + γ 2 √ 2 n ln 2 K δ ) &lt; δ/ 2 . This also suggests that 487

<!-- formula-not-decoded -->

Combining this with (11) and the union bound will complete the proof.

488

Proof of Theorem 3.3. Theorem 3.2 shows that the following holds with probability at least 1 -489 γ -α -δ : 490

<!-- formula-not-decoded -->

Note that 491

<!-- formula-not-decoded -->

Since this determistically holds for all S , combining (13) with (20) completes the proof. 492

## A.1 Approximating the intractable part by a data set 493

Theorem A.1. Given the notations in Theorem 3.1, 494

<!-- formula-not-decoded -->

Proof. Denote n = { n 1 , ..., n K } and for each j ∈ [ K ] : 495

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote y = 4 t uC 2 for any t ∈ [ 0 , uC √ ln γ 8 n -6 ] . The proof for (21) contains three main steps. 496 Step 1: We first observe that 497

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 2 - estimating E S [ e yB K | h , n ] : We observe the following for each j ∈ T S , 498

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore B j = B j -1 + E X j [ X j | h , n ] -X j for all j ∈ T S . Note that B i = B i -1 (due to 499 n i = b i = X i = 0 ) for all i / ∈ T S . Hence, for i / ∈ T S , we will use E X i [ X i | h , n ] -X i instead of 0 500 in the below analysis for simplicity of presentation. 501

We can rewrite 502

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality comes from the fact that X K is conditionally independent with S ≤ K -1 , conditioned on { h , n } .

It is easy to see that 0 ≤ X K ≤ Cn K , due to 0 ≤ F ( S K , h ) ≤ C . Lemma B.1 implies E X K [ e y ( E XK [ X K | h , n ] -X K ) | h , n ] ≤ exp ( y 2 C 2 n 2 K 8 ) . Plugging this into (33), we obtain

503

504

505

506

<!-- formula-not-decoded -->

Using the same arguments for X K -1 , ..., X 1 , we obtain the followings 507

<!-- formula-not-decoded -->

Step 3 - bounding Pr( B K ≥ t ) : By combining this with (26), we obtain 508

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

( Since n K is independent with v 1 , ..., n K -1 )

When γp K &lt; 1 , due to t ≤ uC √ ln γ 8 n -6 , observe that y 2 C 2 8 = 2 t 2 u 2 C 2 ≤ ln γ 4 n -3 ≤ ln γ (1 -γp K )(4 n -3) . Note 509 that n K is a binomial random variable with parameters n and p K . Combining those facts with Lemma 510 B.7 implies E n K exp ( y 2 C 2 8 n 2 K ) ≤ exp ( y 2 C 2 8 γnp K (1 + γnp K ) ) . On the other hand, Lemma B.6 511

also implies E n K exp ( y 2 C 2 8 n 2 K ) ≤ exp ( y 2 C 2 8 γnp K (1 + γnp K ) ) when γp K ≥ 1 . As a result, 512 those facts and (38) lead to the following: 513

<!-- formula-not-decoded -->

Using the same arguments for the remaining variables n K -1 , ..., n 1 , we obtain 514

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As a result 515

<!-- formula-not-decoded -->

Since n j = 0 for all j / ∈ T S , we have 516

<!-- formula-not-decoded -->

Multiplying both sides (of the probability term) with 1 /n leads to 517

<!-- formula-not-decoded -->

518

519

520

521

522

523

Choosing t = C √ u 2 ln 1 δ 1 results in (21), completing the proof.

## B Supporting theorems and lemmas

## B.1 Hoeffding's Lemma

Lemma B.1 (Hoeffding's lemma for conditionals) . Let X be any real-valued random variable that may depend on some random variables Y . Assume that a ≤ X ≤ b almost surely, for some constants a, b . Then, for all λ ∈ R ,

<!-- formula-not-decoded -->

Proof. Denote c = E X [ X | Y ] -b, d = E X [ X | Y ] -a and hence c ≤ 0 ≤ d . 524

Since exp is a convex function, we have the following for all E X [ X | Y ] -X ∈ [ c, d ] : 525

<!-- formula-not-decoded -->

Therefore, by taking the conditional expectation over X for both sides, 526

<!-- formula-not-decoded -->

where L ( h ) = ch d -c +ln(1 + c -e h c d -c ) . For this function, note that 527

<!-- formula-not-decoded -->

The AM-GM inequality suggests that L ′′ ( h ) ≤ 1 / 4 for all h . Combining this property with Taylor's 528 theorem leads to the following, for some θ ∈ [0 , 1] , 529

<!-- formula-not-decoded -->

Combining this with (46) completes the proof. 530

531

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

## B.2 Small random variables

Lemma B.2. Let x 1 , ..., x n be independent random variables in [0 , 1] and satisfy E [ x i ] ≤ ν, ∀ i for some ν ∈ [0 , 1] . For any c ≥ 1 satisfying cν ≥ 1 and any λ ≥ 0 , we have E exp ( λ ( x 1 + · · · + x n ) 2 ) ≤ exp( λcnν (1 + cnν )) .

Lemma B.3. Let x 1 , ..., x n be independent random variables in [0 , 1] and satisfy E [ x i ] ≤ ν, ∀ i for some ν ∈ [0 , 1] . For any c ≥ 1 satisfying cν &lt; 1 and any λ ∈ [0 , ln c (1 -cν )(4 n -3) ] , we have E exp ( λ ( x 1 + · · · + x n ) 2 ) ≤ exp( λcnν (1 + cnν )) .

In order to prove those results, we need the following observations.

Lemma B.4. Consider a random variable X ∈ [0 , 1] with mean E [ X ] ≤ ν for some constant ν ∈ [0 , 1] . For any c ≥ 1 , λ ≥ 0 :

- If cν ≥ 1 , then E e λX ≤ e cνλ .
- If cν &lt; 1 , then E e λX ≤ e cνλ for all λ ∈ [0 , ln c 1 -cν ] .

Proof. The Taylor series expansion of the function e λX at any X is e λX = 1 + ∑ ∞ p =1 ( λX ) p p ! . 543 Therefore 544

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

545

546

547

548

549

550

551

552

553

554

555

556

557

Next we consider function y ( λ ) = e cνλ -1 + ν -νe λ . Its derivative is y ′ = cνe cνλ -νe λ = νe λ ( ce ( cν -1) λ -1) .

For the case cν ≥ 1 , one can observe that y ′ ≥ 0 for all λ ≥ 0 . This means y is non-decreasing, and hence y ( λ ) ≥ y (0) = 0 . As a result, e cνλ ≥ 1 -ν + νe λ ≥ E [ e λX ] .

Consider the case cν &lt; 1 , it is easy to show that y ′ ( λ ) ≥ 0 for all λ ∈ [0 , ln c 1 -cν ] . This means y is non-decreasing in the interval [0 , ln c 1 -cν ] , and hence y ( λ ) ≥ y (0) = 0 for all λ ∈ [0 , ln c 1 -cν ] . As a result, e cνλ ≥ 1 -ν + νe λ ≥ E [ e λX ] , completing the proof.

Corollary B.5. Consider a random variable X ∈ [0 , 1] with mean E [ X ] ≤ ν for some constant ν ∈ [0 , 1] . For all constants a, b ≥ 0 , c ≥ 1 :

- E e λ ( aX 2 + bX ) ≤ e c ( a + b ) νλ , for all λ ≥ 0 , if cν ≥ 1 .

<!-- formula-not-decoded -->

Proof. It is easy to observe that E e λ ( aX 2 ) ≤ E e λ ( aX ) due to X ∈ [0 , 1] . This suggests that E e λ ( aX 2 + bX ) ≤ E e λ ( a + b ) X . Applying Lemma B.4 will complete the proof.

<!-- formula-not-decoded -->

Proof of Lemma B.2. Denote y n = x 1 + · · · + x n . Observe that y n = y n -1 + x n and 558

<!-- formula-not-decoded -->

Since cν ≥ 1 and x n is independent with y n -1 , Corollary B.5 implies E x n e λ (2 x n y n -1 + x 2 n ) ≤ 559 e cνλ (2 y n -1 +1) . Plugging this into (49) leads to 560

<!-- formula-not-decoded -->

Next we consider E y n -1 [ e λ ( y 2 n -1 +2 cνy n -1 ) ] . Observe that y n -1 = y n -2 + x n -1 and hence 561

<!-- formula-not-decoded -->

Since cν ≥ 1 and x n -1 is independent with y n -2 , Corollary B.5 implies 562 E x n -1 e λ (2 x n -1 y n -2 +2 cνx n -1 + x 2 n -1 ) ≤ e cνλ (2 y n -2 +2 cν +1) . Plugging this into (52) leads to 563

<!-- formula-not-decoded -->

By using the same arguments, we can show that 564

<!-- formula-not-decoded -->

=

...

<!-- formula-not-decoded -->

Note that E y 1 [ e λ ( y 2 1 +2 c ( n -1) νy 1 ) ] = E x 1 [ e λ ( x 2 1 +2 c ( n -1) νx 1 ) ] ≤ e cνλ (1+2 c ( n -1) ν ) , according to 565 Corollary B.5. Combining this with (57), we obtain 566

<!-- formula-not-decoded -->

By plugging this into (50), we obtain 567

<!-- formula-not-decoded -->

completing the proof.

Proof of Lemma B.3. Denote y n = x 1 + · · · + x n and observe that

568

569

<!-- formula-not-decoded -->

Note that y n -1 = x 1 + · · · + x n -1 ≤ n -1 and λ (2 y n -1 +1) ≤ λ (2 n -1) ≤ λ (4 n -3) ≤ ln c 1 -cν . 570 Since x n is independent with y n -1 , Corollary B.5 implies E x n e λ (2 x n y n -1 + x 2 n ) ≤ e cνλ (2 y n -1 +1) . 571 Plugging this into (61) leads to 572

<!-- formula-not-decoded -->

Next we consider E y n -1 [ e λ ( y 2 n -1 +2 cνy n -1 ) ] . Observe that 573

e

2

cνλ

(3

cν

+1)

[

e

λ

(

y

2

n

+6

cνy

-

n

3

)

]

(56)

E

y

-

n

3

-

3

<!-- formula-not-decoded -->

One can easily show that λ (2 y n -2 + 2 cν + 1) ≤ λ (2( n -2) + 2 cν + 1) ≤ λ (4 n -574 3) ≤ ln c 1 -cν , since y n -2 = x 1 + · · · + x n -2 ≤ n -2 . Therefore Corollary B.5 implies 575 E x n -1 e λ (2 x n -1 y n -2 +2 cνx n -1 + x 2 n -1 ) ≤ e cνλ (2 y n -2 +2 cν +1) , since x n -1 is independent with y n -2 . 576 Plugging this into (64) leads to 577

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By using the same arguments, we can show that 578

<!-- formula-not-decoded -->

...

<!-- formula-not-decoded -->

Note that E y 1 [ e λ ( y 2 1 +2 c ( n -1) νy 1 ) ] = E x 1 [ e λ ( x 2 1 +2 c ( n -1) νx 1 ) ] ≤ e cνλ (1+2 c ( n -1) ν ) , according to 579 Corollary B.5 and the fact that λ (1 + 2 c ( n -1) ν ) ≤ λ (4 n -3) ≤ ln c 1 -cν . Combining this with (69), 580 we obtain 581

<!-- formula-not-decoded -->

582

583

584

585

586

587

588

589

590

591

592

593

594

595

596

597

By plugging this into (62), we obtain

<!-- formula-not-decoded -->

completing the proof.

## B.3 Binomial and multinomial random variables

Next we analyze some properties of binomial random variables.

Lemma B.6. Consider a binomial random variable z with parameters n ≥ 1 and ν ∈ [0 , 1] . For any c ≥ 1 satisfying cν ≥ 1 and any λ ≥ 0 , we have E e λz 2 ≤ e cnν (1+ cnν ) λ .

Proof. Since z is a binomial random variable, we can write z = x 1 + · · · + x n , where x 1 , ..., x n are i.i.d. Bernoulli random variables with parameter ν . Therefore applying Lemma B.2 completes the proof.

Lemma B.7. Consider a binomial random variable z with parameters n ≥ 1 and ν ∈ [0 , 1] . For any c ≥ 1 satisfying cν &lt; 1 and any λ ∈ [0 , ln c (1 -cν )(4 n -3) ] , we have E e λz 2 ≤ e cnν (1+ cnν ) λ .

Proof. Since z is a binomial random variable, we can write z = x 1 + · · · + x n , where x 1 , ..., x n are i.i.d. Bernoulli random variables with parameter ν . Therefore applying Lemma B.3 completes the proof.

Lemma B.8 (Multinomial variable) . Consider a multinomial random variable ( n 1 , ..., n K ) with parameters n and ( p 1 , ..., p K ) . For any δ &gt; 0 :

<!-- formula-not-decoded -->

Proof. Observe that 598

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequlality can be derived by using the fact that ∑ K i =1 ( 0 . 5 p i + 0 . 5 n i n ) ( p i -n i n ) is a convex combination of the elements in { p i -n i n : i ∈ [ K ] } , because of 1 = ∑ K i =1 ( 0 . 5 p i + 0 . 5 n i n ) . Furthermore, since n i is a binomial random variable with parameters n and p i , Lemma 5 in [23] shows that Pr ( p i -n i n &gt; √ 2 p i n ln K δ ) &lt; δ for all i . This immediately implies Pr ( p i -n i n &gt; √ 2 n ln K δ ) &lt; δ . Combining this fact with (76), we obtain Pr ( ∑ K i =1 p 2 i -∑ K i =1 ( n i n ) 2 &gt; 2 √ 2 n ln K δ ) &lt; δ , completing the proof.

## C Experimental setup

More details about clustering the training images:

- We first preprocessed the images following Pytorch 2 : The images are resized to resize \_ size = [256] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop \_ size = [224] . Finally the values are first rescaled to [0 . 0 , 1 . 0] . Those operations are required for Pytorch pretrained models.
- For each run, we randomly choose 200 points in [0 . 0 , 1 . 0] C × H × W to be the centroids, since each preprocessed image belongs to [0 . 0 , 1 . 0] C × H × W . Those centroids are used to build the small areas Z i in the partition. Each training image x will be assigned to area Z i if it is closest to the centroid of Z i amongst all centroids, according to the Euclidean distance.

2 https://pytorch.org/vision/0.20/models/generated/torchvision.models.vit\_b\_16. html

599

600

601

602

603

604

605

606

607

608

609

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

639

640

641

642

643

644

645

646

647

648

649

650

651

652

653

654

655

656

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

668

669

670

671

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification:

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

Justification:

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

Justification:

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

Justification:

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

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

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification:

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

Justification: Our paper is theoretical, so no special requirement for computer resources is required.

Guidelines:

- The answer NA means that the paper does not include experiments.

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

- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification:

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: At this moment, we do not foresee any negative impact of our work to the world, since our work is theoretical. Nonetheless, we can see positive impacts of our work to deep learning. Although deep learning has been helping us to make many breakthroughs, little has been known about why those DL models can perform really well on unseen data, after training from a finite training set. This is arguably the biggest challenge in DL theory. Our work provides novel theories that are non-vacuous for a large class of modern DL models. Those theories contribute to the solid foundation of DL in particular, and AI in general.

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

Justification:

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification:

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

Justification:

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

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

937

938

939

940

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification:

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
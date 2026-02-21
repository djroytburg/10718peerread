25

## Spatial Discriminability of CLIP for Training-Free Open-Vocabulary Semantic Segmentation

## Anonymous Author(s)

Affiliation Address email

## Abstract

Extending CLIP models to semantic segmentation remains a considerable challenge, largely due to the misalignment between their image-level pre-training objectives and the pixel-level spatial understanding required for dense predictions. Prior efforts have achieved encouraging results by reorganizing the final layer and feature representations of CLIP to enhance dense predictions. However, these approaches often inherit the global alignment bias of the final layer, leading to suboptimal spatial discriminability and segmentation performance. In this work, we propose TLH-CLIP, a novel training-free framework that systematically exploits the spatial discriminability across Token , Layer and Head levels in CLIP for dense predictions. Through comprehensive analysis, we uncover three key findings: (i) some anomalous tokens emerges in the final layers, which are category-agnostic but disproportionately attract attention from semantically meaningful patch tokens, thereby degrading spatial discriminability; (ii) the final few layers primarily enhance global image-text alignment with great sacrifice of local discriminability (e.g., last 3 layers in ViT-B-16 and 5 layers in ViT-L-14); (iii) a few attention heads (e.g., 10 out of 144 in ViT-B/16) demonstrate strong spatial discriminability across different datasets. Motivated by these insights, we propose three complementary techniques: abnormal token replacement, semantic-spatial reweighting, and selective head enhancement to effectively recover spatial coherence and improve segmentation performance without any additional training, auxiliary pre-trained networks, or extensive hyperparameter tuning. Extensive experiments on 8 common semantic segmentation benchmarks demonstrate that TLH-CLIP achieves state-of-the-art performance across diverse scenarios, highlighting its effectiveness and practicality for real-world deployment.

## 1 Introduction

Recent advances in vision-language pretrained models, such as CLIP [1], have demonstrated remark26 able generalization and open-vocabulary recognition capabilities at the image level, thereby opening 27 up possibilities for transferring image-text alignment to pixel-level tasks. Despite this progress, they 28 often underperform in dense prediction tasks like semantic segmentation, primarily due to their 29 limited capacity to localize fine-grained visual details [2, 3]. To address these limitations, several 30 studies have incorporated trainable modules into CLIP, typically relying on additional forms of 31 supervision such as dense annotations for a restricted set of categories [4, 5, 6, 7] or supplementary 32 image-text pairs [8, 9, 10, 11, 12]. Although these approaches have demonstrated improved seg33 mentation performance, they incur significant computational and annotation costs. Furthermore, the 34 dependence on limited supervision often undermines the generalizability of the model, making it 35 prone to overfitting the training distribution. 36

These challenges have sparked increasing interest in training-free methods[3, 13, 14, 15, 16, 17, 18, 37 19, 20], which aim to adapt CLIP's pre-trained representations for semantic segmentation without 38 additional training, while preserving its generalization capability. A key difficulty in this direction is 39 enhancing spatial representations for accurate pixel-level predictions. For instance, MaskCLIP[14] 40 computes similarity between key features in the final attention layer to enrich patch embeddings. 41 SCLIP [3] replaces the standard query-key attention with correlative self-attention (query-query 42 and key-key). ClearCLIP [15] further removes residual connections and discards the FFN in the 43 final layer to reduce noise and improve spatial alignment. ResCLIP [20] incorporates attention 44 maps from earlier layers to refine final-layer attention map. However, these methods largely focus 45 on modifying the final-layer attention, often leading to suboptimal ambiguous local relationships 46 and noisy segmentation. To address spatial limitations, some approaches incorporate features from 47 auxiliary backbones such as DINO [21, 17], SAM [17, 22], or diffusion models [23, 24]. While 48 effective, these methods incur significant computational and memory overhead. 49

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

Motivated by these limitations, we begin with a layer-wise analysis of spatial discriminability and text-semantic alignment within the CLIP model. As shown in Figure 1, we observe a clear spatialsemantic trade-off in the final layers: spatial discriminability drops sharply, while the improvement in semantic alignment is relatively marginal. To understand the cause of this phenomenon, we further examine internal token interactions and structural patterns across layers. Through attention map visualizations, we find that certain abnormal tokens emerge in the deeper layers, attracting disproportionately high attention from nearly all spatial positions. This behavior causes the majority of tokens to converge on a small subset, thereby disrupting the spatial coherence of the representation. Further analysis reveals that these abnormal tokens exhibit sparse and high-magnitude activations. Moreover, they are class-agnostic, as their similarity remains consistent across different positions, layers, and input samples, indicating a lack of semantic specificity. Contrary to prior assumptions that such tokens encode global semantic content, our findings suggest they may instead function as bias components that offset global-mean features, thereby facilitating alignment with text embeddings.

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

78

79

80

81

82

83

84

85

86

Based on the analysis, we propose TLH-CLIP, a training-free framework that leverages the inherent properties of CLIP to enhance the spatial discriminability of visual features while preserving their semantic alignment. TLH-CLIP comprises three complementary strategies: abnormal token replacement (ATR), spatial-semantic reweighting (SSR), and selective head enhancement (SHE). Specifically, the ATR employs hoyer scores to identify abnormal tokens by thresholding their characteristic sparsity. Once detected, these anomalous tokens are replaced with a weighted average of normal tokens, based on spatial distance. To mitigate the degradation of spatial discriminability in the earlier final layers, SSR reweights the contributions of the residual pathway relative to the attention and FFN submodules. This adjustment restores a better balance between spatial coherence and semantic abstraction, leveraging the fact that late-intermediate layers exhibit stronger spatial discriminability while maintaining comparable levels of semantic alignment. Finally, SHE further enhances spatial coherence by selectively aggregating features from attention heads with high spatial discriminability, using them to refine the output representations. Experimental results demonstrate that TLH-CLIP achieves significant performance improvements when integrated into various baseline methods, establishing new state-of-the-art results across eight benchmark datasets.

Contributions. Our contributions can be summarized as follows:

- We conduct a comprehensive analysis of spatial discriminability at the token, head, and layer levels.
- We propose TLH-CLIP, a novel training-free approach, terms TLH-CLIP. To the best of our knowledge, this is the first work to explicitly modify the inference procedure prior to the final layer, enabling improved spatial coherence without compromising semantic alignment.
- The extensive experiment results on open-vocabulary semantic segmentation tasks consistently demonstrate the effectiveness of the proposed method.

## 2 Analysis

## 2.1 Preliminaries

CLIP employs a Vision Transformer (ViT) [25] as its image encoder to generate visual representations 87 that are aligned with corresponding textual descriptions. The vision encoder first tokenizes an input 88 image of size H × W × 3 by dividing it into a grid of non-overlapping patches of size P × P , 89

yielding h = H/P rows and w = W/P columns of patches. Each patch is then linearly projected 90 into a D -dimensional embedding space, x i ∈ R D , and augmented with positional embeddings. An 91 additional learnable [CLS] token is prepended to the sequence and is later used for image-level 92 prediction. The resulting token sequence is denoted as X 0 = [ x 0 cls , x 0 1 , . . . , x 0 hw ] ∈ R (1+ hw ) × D . This 93 sequence is passed through a stack of L Transformer encoder layers, each consisting of a multi-head 94 self-attention (MSA) module followed by a feed-forward network (FFN). Let LN denotes layer 95 normalization, the token representations are updated at each layer l as follow: 96

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The CLIP model is originally trained on large-scale image-text pairs for open-vocabulary image recognition tasks. To extend it to semantic segmentation, a natural approach is to compute the similarity between the visual tokens X L = [ x L 1 , . . . , x L hw ] from the final Transformer layer and the textual embeddings of C category names, denoted by t ∈ R C × D . This results in a patch-text similarity map of size hw × C . Denote t c as the embedding of the c -th class name, the final segmentation prediction is obtained by applying an argmax operation over the class dimension of this similarity map, as follows:

<!-- formula-not-decoded -->

Ideally, for effective semantic segmentation, the vision encoder should produce feature representations that satisfy two key properties:

- Spatial discriminability (SD) : token features should exhibit high internal consistency within the same semantic category while remaining clearly distinguishable from those of other categories, thereby enabling accurate and clean segmentation results.
- Semantic alignment (SA) : token features should be well-aligned with their corresponding textual embeddings to enable semantically meaningful segmentation results.

Beyond their importance in open-vocabulary semantic segmentation, these two properties are also more highly relevant to the development of multimodal large language models (MLLMs), as the vision encoder of CLIP is often directly employed to extract visual representations without additional training, serving as input to downstream language models such as LLaVA [26, 27]. In this work, we aim to enhance the spatial discriminability of CLIP features in a training-free manner, thereby preserving the its strong generalization capability.

## 2.2 Analysis of layer-wise spatial discriminability and semantic alignment

Significant decline in SD with marginal gains in SA in the final layers. To assess whether CLIP visual features exhibit the desired properties, we investigate the layer-wise SD and SA within CLIP models. To quantitatively assess SD property, we follow the evaluation protocol proposed in [28]. In particular, we extract patch-level feature representations from the vision encoder for each image and associate them with corresponding semantic labels using the ground-truth segmentation masks from Pascal VOC [29], PASCAL Context [30], ADE20K [31], and COCO-Stuff [32] datasets. Specifically, let x l i ∈ R D and x l j ∈ R D denote the feature representations of two image patches i and j extracted from the l -th layer of the encoder. These feature vectors are ℓ 2 -normalized, and their cosine similarity is computed to serve as the prediction of a binary classifier that indicates whether the two patches belong to the same semantic category. Given the corresponding semantic labels t ( x i ) and t ( x j ) , the target value for classification is set to 1 if t ( x i ) = t ( x j ) , and 0 otherwise. To evaluate the SA property, we extract the intermediate representations x l i ∈ R D from each individual visual token at layer l , and use them as inputs to the final layer to project these features into the final visual latent space for semantic prediction. Following [15], we remove the FFN and residual connections in the final layer to avoid introducing contaminating semantic information. Additionally, inspired by [14], we replace the last-layer attention matrix with an identity matrix to avoid noisy integration during the final attention computation. The final visual representation of each layers can be expressed as v l i = x l i W L v W L o ∈ R D , where W L v and W L o denotes the value and output project matrix in last-layer MSA module. Based on these representations, SA is measured using the average accuracy between the predicted and ground-truth semantic labels, following Equation (3).

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

132

133

134

135

136

137

Figure 1: Layer-wise analysis of spatial discriminability (blue curves) and semantic alignment (orange curves) within the CLIP vision encoders across different datasets. The final layer is excluded from the analysis to avoid discrepancies caused by prior modifications to the last-layer in different methods.

<!-- image -->

Figure 2: Visualization of the abnormal token phenomenon in the attention maps across different layers of the ViT-B/16 model in the CLIP vision encoder.

<!-- image -->

We present the layerwise SD and SA scores for both the ViT-B/16 and ViT-L/14 models used as the 138 CLIP vision encoder in Figure 1. From the results, we make the following observations: 139

140

141

142

143

- The SD of CLIP exhibits an inverted U-shaped curve across layers: it initially increases in the early stages but declines in the deeper layers. This decline is especially prominent in the final layers. For example, the last two layers ( (excluding the final layer)) of the ViT-B/16 model and the last five layers of the ViT-L/14 model show a marked reduction in spatial discriminability.

144

145

- SA follows an approximately monotonic increasing pattern across layers: it improves substantially in the early layers but gradually saturates in the final layers, offering only marginal gains thereafter.

These findings offer a nuanced understanding of why CLIP has proven effective for open-vocabulary 146 semantic segmentation. In particular, the strong semantic alignment observed in the final layers 147 explains why prior work often leverages last-layer features for aligning visual tokens with textual 148 categories. However, the significant decline in spatial discriminability in these layers reveals a key 149 limitation as they may lack the fine-grained spatial distinctions necessary for producing accurate and 150 precise segmentation masks. In this work, we aim to address this limitation by proposing methods 151 that jointly preserve spatial structure and semantic alignment through a systematic exploitation of 152 spatial discriminability across token, layer, and head levels. Before introducing our approach, we first 153 investigate the underlying causes of the decline in spatial discriminability in the next subsection. 154

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

## 2.3 Analysis of abnormal tokens

Class-agnostic sparse and large-norm tokens. To understand the progression within the vision encoder, we analyze attention maps across layers. As shown in Figure 2, deeper layers exhibit a small set of dominant tokens that receive disproportionately high attention from nearly all spatial locations, causing most tokens to focus on this subset, consistent with prior observations [33, 18].This leads to a gradual decline in spatial discriminability, which is essential for accurate segmentation. To further characterize these dominant tokens, we compare their features with those of normal tokens. As illustrated in Figure 3, dominant tokens exhibit sparse and consistent activation patterns, with only a few channels maintaining high activation. To quantify this sparsity, we adopt the hoyer score [34]:

<!-- formula-not-decoded -->

where x l i ∈ R D is the feature vector of the i -th token at layer l . We use this metric to quantify sparsity and visualize its distribution across layers and token positions in Figure 3(b). To evaluate whether dominant tokens encode meaningful semantics, we analyze their pairwise cosine similarity across spatial locations, layers, and image samples on the ImageNet validation set. As shown in Figure 4, these tokens exhibit strong invariance across positions and inputs, indicating limited semantic specificity. Contrary to prior assumptions that they capture global semantic content, our results suggest they act more like bias components that offset global-mean features, facilitating text alignment, similar to the bias term in final-layer classifiers under neural collapse [35, 36].

## 3 Method

In this section, we provide a detailed description of our training-free framework, which comprises three components: Abnormal Token Replacement (ATR) in Section 3.1, Spatial-Semantic Reweighting (SSR) in Section 3.2, and Selective Head Enhancement (SHE) in Section 3.3. Each component is complementary, and together they work synergistically to enhance the spatial discriminability of the CLIP model, based on our previous analysis.

## 3.1 Abnormal token replacement (ATR)

To mitigate the adverse effects of these anomalous tokens, we propose a simple yet effective strategy to suppress their influence prior to the final layer. As demonstrated in our earlier analysis, these tokens exhibit characteristically sparse activation patterns. To systematically identify them, we employ the hoyer score H ( x l i ) defined before as a sparsity-based criterion. Tokens with scores exceeding a predefined threshold τ are deemed anomalous and grouped into the set A l = { i |H ( x l i ) &gt; τ } . After identifying them, we suppress their influence using an unnormalized 2-dimensional Gaussian kernel. Specifically, each anomalous token at spatial position ( m,n ) ∈ A is replaced by a weighted aggregation of its neighboring non-anomalous tokens:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, σ controls the spatial extent of smoothing, and the weights w m,n,i,j ensure that only normal 187 tokens contribute to the reconstruction of anomalous ones. Empirically, we find that applying this 188 strategy before the penultimate layer leads to a performance drop, likely due to the removal of inherent 189 biases encoded in abnormal tokens, which substantially alters the inference process. Therefore, we 190 apply it only at the penultimate layer, i.e., with l = L -1 . 191

192

193

194

## 3.2 Spatial-semantic reweighting (SSR)

After mitigating the influence of anomalous tokens in the input to the last layer, the model exhibits improved spatial discriminability. However, a critical challenge remains: anomalous tokens present

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

Figure 3: Illustration of the sparsity and high-norm characteristics of abnormal tokens. Figure (a) shows the attention map of the red anchor token. Figure (b) presents the Hoyer score distribution across layers and spatial positions. Figures (c)-(e) depict the channel activations of a normal token (red) and two abnormal tokens (yellow and blue) highlighted in Figure (a).

<!-- image -->

Figure 4: Layer-wise cosine similarity among abnormal tokens across positions, layers and samples.

<!-- image -->

in earlier layers may have already disrupted the spatial coherence of feature representations, limiting 195 the effectiveness of final-layer refinements. Based on our layer-wise analysis, the final few layers 196 overly emphasize alignment with text embeddings, the marginal gains in semantic alignment come at 197 the cost of a pronounced decline in spatial discriminability. To address this imbalance, we propose a 198 spatial-semantic reweighting strategy that enhances the model's spatial awareness while preserving 199 its semantic alignment capabilities. Given the feature representation X l -1 at the l -th layer within 200 the final few layers (e.g., layers 10-11 in ViT-B/16 and layers 20-23 in ViT-L/14), we reweight 201 the forward pass by upweighting the residual pathway and downweighting the attention and MLP 202 submodules, as follows: 203

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where α ∈ [0 , 1] is a reweighting coefficient that controls the relative degree of emphasis on the residual signal. As α increases, the l -th block increasingly preserves spatially discriminative features from earlier layers via the residual pathway, while diminishing the dominant influence of semantic aggregation in the attention and MLP submodules. To the best of our knowledge, prior work has primarily focused on reforming the final layer or modifying its representations to improve performance. However, these approaches often inherit the global semantic alignment bias inherent in the final few layers, resulting in a substantial decline in the spatial discriminability of the extracted features. In contrast, our SSR strategy explicitly mitigates this limitation by rebalancing the contributions of residual and semantic components in intermediate layers preceding the final layer.

## 3.3 Selective head enhancement (SHE)

Strong spatial discriminability of some attention heads. While the proposed strategies effectively enhance the spatial discriminability in the final layers, the overall spatial discriminability of the features output by the CLIP vision encoder may still remain suboptimal. Inspired by recent studies [37, 38] revealing that different attention heads capture distinct visual concepts, such as number, shape and texture, this motivates us to investigate whether certain heads are specifically responsible for encoding spatial discriminability. To identify such heads, we follow the formulation introduced in [39, 37], which rewrites the multi-head self-attention (MSA) output as a summation over H independent attention heads: Attn ( LN ( X l )) = ∑ H h =1 A l h V l h W l o ∈ R (1+ hw ) × D , where A l h and V l h denote the attention and value matrices for the h -th head at layer l , and W l o is the output projection matrix shared across all heads. We extract the contribution of the h -th head at layer l and apply abnormal token resolution as follows:

<!-- formula-not-decoded -->

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

Figure 5: Head-wise analysis of spatial discriminability within the ViT-B/16 vision backbone across multiple datasets. To ensure consistency, the final layer is excluded, and only the top 50 attention heads are visualized in each figure for clarity.

<!-- image -->

where

σ

(

·

)

denotes the abnormal token replacement operation defined previously. To assess the SD

of each attention head, we adopt the same AUC-based metric as the preceding layer-level analysis.

Figure 5 shows the head AUC distribution for ViT-B/16, with ViT-L/14 results in the appendix. From the figure, we observe that the output features from certain attention heads, such as the 9th head in

the 8th layer, consistently exhibit high AUC scores across different datasets, suggesting that these heads are more effective at capturing SD information than others.

Building on this observation, we propose to selectively leverage high-performing heads to enhance the spatial discriminability of the output representations. Let AUC s l,h denote the AUC score of the representations from the h -th head in the l -th layer for dataset s ∈ { VOC , Context , ADE , Stuff } . To obtain a dataset-agnostic measure of discriminability, we compute the average AUC score for each head across all datasets, denoted as AUC l,h . The distribution of these average scores is provided in the appendix. We rank all heads by their AUC l,h scores and select the topk to form the set H top k . The corresponding feature representations are then aggregated as: X top k = 1 k ∑ ( l,h ) ∈H top k X l,h . This aggregated feature X top k is used to construct a similarity map S = X top k X ⊤ top k ∥ X top k ∥ 2 , which captures the pairwise similarity among visual tokens. To mitigate the influence of spurious interactions between tokens from different semantic categories, we apply a thresholding operation with a predefined parameter β , resulting in the filtered similarity map S β , where S β ( i, j ) = S ( i, j ) if S ( i, j ) ≥ β , and S β ( i, j ) = 0 otherwise. The resulting S β is then column-wise normalized, and subsequently used to refine the final-layer features by X L -1 = Norm ( S β ) X L -1 .

## 4 Experiment Results

Evaluation datasets. We follow the standard evaluation protocol from prior works [3, 15, 16] and assess our method on eight widely used semantic segmentation benchmarks. For clarity, we group them into two categories and use abbreviated names throughout the paper. The first category excludes background and includes Pascal VOC [29] (VOC20), Pascal Context [30] (Context59), COCO-Stuff [32] (Stuff), ADE20K [31] (ADE), and Cityscapes [40] (City). The second includes background and consists of VOC21, Context60, and COCO-Object [32] (Object). We use CLIP [1] models with ViT-B/16 and ViT-L/14 backbones via MMSegmentation [41], and report results using the mean Intersection-over-Union (mIoU). All hyperparameters are fixed across datasets without task-specific tuning. Additional implementation details are provided in the appendix.

## 4.1 Comparison with existing methods.

We compare our approach against a comprehensive set of open-vocabulary semantic segmentation (OVSS) methods, including the direct baseline CLIP [1], as well as several state-of-the-art trainingfree approaches: MaskCLIP [14], CLIPSurgery [13], SCLIP [3], NACLIP [16], ClearCLIP [15], LAVG [42], and ResCLIP [20]. We also include several influential weakly supervised methods, such as GroupViT [5], ReCo [43], and TCL [8]. Unless otherwise specified, all reported results are taken directly from the respective original papers and ResCLIP [20]. As our method is orthogonal to approaches that primarily target improvements in the final-layer attention, we additionally evaluate its effectiveness when integrated with recent state-of-the-art methods that employ specialized attention mechanisms in the last layer, including SCLIP [3], ClearCLIP [15], and ResCLIP [20]. For fair

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

292

293

294

295

296

297

298

Table 1: Performance comparison of our approach with other methods on eight semantic segmentation benchmarks following the evaluation protocol in Section 4. Our results are marked in gray.

| Methods          | Training   | With a background class   | With a background class   | With a background class   | Without background class   | Without background class   | Without background class   | Without background class   | Without background class   | Avg.        |
|------------------|------------|---------------------------|---------------------------|---------------------------|----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|-------------|
| Methods          | Training   | VOC21                     | Context60                 | Object                    | VOC20                      | City                       | Context59                  | ADE                        | Stuff                      | Avg.        |
| ReCo [43]        | ✓          | 25.1                      | 19.9                      | 15.7                      | 57.7                       | 21.1                       | 22.3                       | 11.2                       | 14.8                       | 23.5        |
| GroupViT [5]     | ✓          | 52.3                      | 18.7                      | 27.5                      | 79.7                       | 18.5                       | 23.4                       | 10.4                       | 15.3                       | 30.7        |
| TCL [8]          | ✓          | 51.2                      | 24.3                      | 30.4                      | 77.5                       | 23.1                       | 30.3                       | 14.9                       | 19.6                       | 33.9        |
| CLIP [1]         | ✗          | 16.2                      | 7.7                       | 5.5                       | 41.8                       | 5.5                        | 9.2                        | 2.1                        | 4.4                        | 11.6        |
| MaskCLIP [14]    | ✗          | 38.8                      | 23.6                      | 20.6                      | 74.9                       | 16.4                       | 26.4                       | 9.8                        | 14.8                       | 28.2        |
| CLIPSurgery [13] | ✗          | 55.2                      | 18.7                      | 27.5                      | 79.7                       | 18.5                       | 23.4                       | 10.4                       | 15.3                       | 31.1        |
| LaVG [42]        | ✗          | 62.1                      | 31.6                      | 34.2                      | 82.5                       | 26.2                       | 34.7                       | 15.8                       | 23.2                       | 38.8        |
| NACLIP [16]      | ✗          | 58.9                      | 32.2                      | 33.2                      | 79.7                       | 35.5                       | 35.2                       | 17.4                       | 23.3                       | 39.4        |
| SCLIP [3]        | ✗          | 59.7                      | 31.7                      | 33.5                      | 81.5                       | 32.3                       | 34.5                       | 16.5                       | 22.7                       | 39.1        |
| +TLH-CLIP (ours) | ✗          | 64.8                      | 34.8                      | 36.6                      | 86.3                       | 36.1                       | 37.6                       | 18.0                       | 24.9                       | 42.4 (+3.3) |
| ClearCLIP [15]   | ✗          | 57.0                      | 32.2                      | 32.5                      | 82.3                       | 32.8                       | 35.8                       | 17.3                       | 24.0                       | 39.2        |
| +TLH-CLIP (ours) | ✗          | 63.9                      | 35.2                      | 35.6                      | 85.7                       | 37.8                       | 38.8                       | 19.2                       | 25.8                       | 42.7 (+3.5) |
| ResCLIP [20]     | ✗          | 60.0                      | 32.7                      | 34.0                      | 85.5                       | 35.6                       | 35.8                       | 17.7                       | 23.8                       | 40.6        |
| +TLH-CLIP (ours) | ✗          | 63.9                      | 35.5                      | 35.3                      | 86.8                       | 38.2                       | 38.2                       | 19.1                       | 25.5                       | 42.8 (+2.2) |

comparison, we exclude the Semantic Feedback Refinement module in ResCLIP, as it relies on the computationally expensive PAMR [44] post-processing, which is inconsistent with our evaluation setting. For comprehensiveness, results on the ViT-L/14 architecture are provided in the appendix.

In Table 1, we summarize the performance of various open-vocabulary semantic segmentation models on benchmark datasets using the ViT-B/16 backbone. Our proposed TLH-CLIP consistently enhances the performance of state-of-the-art approaches, including SCLIP [3], ClearCLIP [15], and ResCLIP [20]. Notably, when integrated with ResCLIP [20], TLH-CLIP achieves state-of-the-art results, outperforming leading weakly supervised methods. As a plug-and-play solution, TLH-CLIP yields consistent improvements across all datasets compared to the respective baselines, demonstrating its strong generalization capability. We further evaluate performance on the ViT-L/14 backbone. In line with observations from [20], existing methods generally exhibit a performance drop exceeding 2% mIoU when adapting to a different backbone; for instance, ClearCLIP [15] suffers a notable decline of 2.7% mIoU. In contrast, when augmented with TLH-CLIP, this performance degradation is significantly alleviated, highlighting the robustness of our approach. Across both backbones, TLH-CLIP delivers substantial improvements over baseline methods, validating its effectiveness.

## 4.2 Experimental analysis

In this section, we conduct comprehensive ablation studies to validate the effectiveness of our proposed method. We adopt SCLIP [3] as the baseline, which enhances spatial correlation by modifying the attention mechanism in the final layer, replacing the standard QK ⊤ attention with a combination of QQ ⊤ + KK ⊤ . In addition, following prior work [15, 20], we remove the residual connections and FFN from the final transformer block to isolate the impact of attention refinement.

Analysis of the hoyer threshold parameter τ . Our method relies on hoyer sparsity to identify anomalous tokens, making the sparsity threshold τ a critical hyperparameter. We conduct a systematic evaluation, as shown in Table 2. At τ = 0 . 2 , many normal tokens are misclassified, leading to excessive smoothing and degraded performance. As τ increases to 0.4, performance steadily improves, but plateaus between 0.5 and 0.8, with a decline observed beyond this range. The broad stable region indicates a clear sparsity gap between normal and abnormal tokens, highlighting the robustness of ATR to threshold selection. Based on this analysis, we fix τ = 0 . 5 for all experiments.

Analysis of spatial-semantic reweighting parameters and number of Layers. To evaluate the impact of the reweighting strength α and the range of layers involved, from l start to l end, we perform a comprehensive sensitivity analysis. The results are summarized in Table 3. We observe that the best performance is obtained when reweighting is applied to layers 10-11 in the ViT-B/16 backbone. This aligns with our earlier findings that these layers experience a marked decline in spatial discriminability while yielding only marginal improvements in semantic alignment. Extending reweighting to include layer 9 results in a slight gain in spatial discriminability but introduces noisy

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

325

326

327

328

329

330

331

332

| Table 2: Study of hoyer sparsity threshold   | Table 2: Study of hoyer sparsity threshold   | Table 2: Study of hoyer sparsity threshold   | Table 2: Study of hoyer sparsity threshold   | Table 2: Study of hoyer sparsity threshold   | Table 2: Study of hoyer sparsity threshold   | Table 2: Study of hoyer sparsity threshold   | Study of ( l start , l end ,α ) in SSR module.   | Study of ( l start , l end ,α ) in SSR module.   | Study of ( l start , l end ,α ) in SSR module.   | Study of ( l start , l end ,α ) in SSR module.   | Study of ( l start , l end ,α ) in SSR module.   |
|----------------------------------------------|----------------------------------------------|----------------------------------------------|----------------------------------------------|----------------------------------------------|----------------------------------------------|----------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| τ                                            | C60                                          | Obj                                          | C59                                          | City                                         | Avg                                          | ( l start                                    | C60 Obj                                          |                                                  | C59                                              | City                                             | Avg                                              |
| τ = 0 . 2                                    | 0.8                                          | 2.0                                          | 1.5                                          | 1.7                                          | 1.5                                          |                                              | baseline 32.4                                    | 32.9                                             | 36.0                                             | 34.3                                             | 33.9                                             |
| τ = 0 . 4                                    | 32.8                                         | 34.0                                         | 36.6                                         | 34.7                                         | 34.5                                         | (9, 11, 0.1)                                 | 32.7                                             | 32.0                                             | 36.5                                             | 36.7                                             | 34.5                                             |
| τ = 0 . 5                                    | 32.8                                         | 34.2                                         | 36.7                                         | 34.7                                         | 34.6                                         | (10, 11, 0.1)                                | 33.1                                             | 33.4                                             | 36.9                                             | 35.6                                             | 34.8                                             |
| τ = 0 . 8                                    | 32.8                                         | 33.9                                         | 36.6                                         | 34.7                                         | 34.5                                         | (11, 11, 0.1)                                | 32.7                                             | 34.1                                             | 36.4                                             | 34.9                                             | 34.5                                             |
| τ = 0 . 9                                    | 32.8                                         | 33.9                                         | 36.6                                         | 34.3                                         | 34.4                                         | (10, 11, 0.05)                               | 32.8                                             | 33.7                                             | 36.4                                             | 35.0                                             | 34.5                                             |
| baseline                                     | 32.4                                         | 32.9                                         | 36.0                                         | 34.3                                         | 33.9                                         | (10, 11, 0.2)                                | 32.6                                             | 31.7                                             | 36.5                                             | 36.6                                             | 34.4                                             |
| Table 4: Study of number of selected         | Table 4: Study of number of selected         | Table 4: Study of number of selected         | Table 4: Study of number of selected         | Table 4: Study of number of selected         | Table 4: Study of number of selected         | Table 5: Combination of three strategies.    | Table 5: Combination of three strategies.        | Table 5: Combination of three strategies.        | Table 5: Combination of three strategies.        | Table 5: Combination of three strategies.        | Table 5: Combination of three strategies.        |
| k                                            | C60                                          | Obj                                          | C59                                          | City                                         | Avg                                          | Methods                                      | Module                                           | Module                                           | Module                                           | mIoU                                             | ∆                                                |
| baseline                                     | 32.8                                         | 34.2                                         | 36.7                                         | 34.7                                         | 34.6                                         |                                              | ATR                                              | SSR                                              | SHE                                              |                                                  |                                                  |
| layer( l = 8 )                               | 33.9                                         | 37.1                                         | 37.1                                         | 35.0                                         | 35.8                                         |                                              |                                                  |                                                  |                                                  |                                                  |                                                  |
| k = 1                                        | 33.4                                         | 37.1                                         | 36.6                                         | 35.4                                         | 35.3                                         | baseline                                     | -                                                | -                                                | -                                                | 33.9                                             | -                                                |
| k = 10                                       | 34.8                                         | 37.6                                         | 37.9                                         | 36.3                                         | 36.7                                         |                                              | ✓                                                | ✓                                                | -                                                | 35.3                                             | +1.4                                             |
| k = 30                                       | 34.7                                         | 37.3                                         | 37.9                                         | 36.4                                         | 36.6                                         |                                              | ✓                                                | -                                                | ✓                                                | 36.7                                             | +2.8                                             |
| k = 50                                       | 34.7                                         | 37.3                                         | 37.8                                         | 36.3                                         | 36.5                                         | Ours                                         | ✓                                                | ✓                                                | ✓                                                | 37.4                                             | +3.5                                             |

semantic signals, ultimately leading to a reduction in segmentation performance. In addition, we examine the effect of varying the reweighting threshold parameter α . As α increases from 0 to 0.1, performance improves steadily, indicating a beneficial balance between spatial and semantic cues. However, further increasing α leads to a performance drop, as it incorporates more noisy semantic information from earlier layers and significantly perturbs the input distribution of subsequent layers.

Analysis of the number of selected heads. We study the effect of varying the number of topk attention heads selected for enhancement, as shown in Table 4. Empirically, we find that SHE is most effective when combined with ATR; without ATR, the spatially coherent similarity maps can cause normal tokens to be fused with abnormal ones. Therefore, we adopt the baseline SCLIP model equipped with ATR as our baseline. On the ViT-B/16 backbone, increasing k from 1 to 10 improves segmentation accuracy, as aggregating multiple spatially discriminative heads helps suppress spurious correlations. However, performance declines when k becomes too large due to the inclusion of noisy or less informative heads, which introduce undesired cross-category interactions. We also compare head- and layer-level selection (best l = 8 ), finding that head-level selection consistently performs better, as discriminative heads are distributed across layers, while entire-layer selection introduces irrelevant heads and degrades performance.

Study of each individual components In the previous parts, we evaluated the effectiveness of each individual component. Table 5 presents their combinations, which yield a substantial improvement of 3.5 mIoU, achieving a final mIoU of 37.5 on these four datasets. These results highlight the complementary contributions of each module to the overall segmentation performance.

## 5 Conclusion

In this paper, we present a comprehensive analysis of the spatial discriminability of pretrained CLIP models across the token, layer, and head levels. Our study reveals three key findings: (1) the emergence of class-agnostic abnormal tokens with sparse, high-norm activations; (2) a notable decline in spatial discriminability in the final layers, despite marginal gains in semantic alignment; and (3) consistently strong spatial discriminability in specific attention heads. Motivated by these observations, we propose TLH-CLIP, a training-free framework that enhances spatial discriminability while preserving semantic alignment. TLH-CLIP introduces three complementary components: (1) abnormal token replacement, (2) spatial-semantic reweighting, and (3) selective head enhancement. Unlike prior methods that focus on modifying the final attention layer, our approach provides lightweight, plug-and-play modules compatible with existing architectures. Extensive experiments on multiple segmentation benchmarks demonstrate that TLH-CLIP consistently outperforms strong baselines. Moreover, as CLIP vision encoders are often frozen during the training of MLLMs, our findings offer valuable insights for improving visual understanding in broader MLLMs.

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

378

379

380

381

382

383

384

## References

- [1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning , pages 8748-8763. PmLR, 2021.
- [2] Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong Tang, Zheng Zhu, Guan Huang, Jie Zhou, and Jiwen Lu. Denseclip: Language-guided dense prediction with context-aware prompting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 18082-18091, 2022.
- [3] Feng Wang, Jieru Mei, and Alan Yuille. Sclip: Rethinking self-attention for dense vision-language inference. In European Conference on Computer Vision , pages 315-332. Springer, 2024.
- [4] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. A simple baseline for open-vocabulary semantic segmentation with pre-trained vision-language model. In European Conference on Computer Vision , pages 736-753. Springer, 2022.
- [5] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. Groupvit: Semantic segmentation emerges from text supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 18134-18144, 2022.
- [6] Yun Xing, Jian Kang, Aoran Xiao, Jiahao Nie, Ling Shao, and Shijian Lu. Rewrite caption semantics: Bridging semantic gaps for language-supervised semantic segmentation. Advances in Neural Information Processing Systems , 36:68798-68809, 2023.
- [7] Yongkang Li, Tianheng Cheng, Bin Feng, Wenyu Liu, and Xinggang Wang. Mask-adapter: The devil is in the masks for open-vocabulary segmentation. arXiv preprint arXiv:2412.04533 , 2024.
- [8] Junbum Cha, Jonghwan Mun, and Byungseok Roh. Learning to generate text-grounded mask for openworld semantic segmentation from only image-text pairs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 11165-11174, 2023.
- [9] Huaishao Luo, Junwei Bao, Youzheng Wu, Xiaodong He, and Tianrui Li. Segclip: Patch aggregation with learnable centers for open-vocabulary semantic segmentation. In International Conference on Machine Learning , pages 23033-23044. PMLR, 2023.
- [10] Pengzhen Ren, Changlin Li, Hang Xu, Yi Zhu, Guangrun Wang, Jianzhuang Liu, Xiaojun Chang, and Xiaodan Liang. Viewco: Discovering text-supervised segmentation masks via multi-view semantic consistency. arXiv preprint arXiv:2302.10307 , 2023.
- [11] Jilan Xu, Junlin Hou, Yuejie Zhang, Rui Feng, Yi Wang, Yu Qiao, and Weidi Xie. Learning openvocabulary semantic segmentation models from natural language supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 2935-2944, 2023.
- [12] Fei Zhang, Tianfei Zhou, Boyang Li, Hao He, Chaofan Ma, Tianjiao Zhang, Jiangchao Yao, Ya Zhang, and Yanfeng Wang. Uncovering prototypical knowledge for weakly open-vocabulary semantic segmentation. Advances in Neural Information Processing Systems , 36:73652-73665, 2023.
- [13] Yi Li, Hualiang Wang, Yiqun Duan, and Xiaomeng Li. Clip surgery for better explainability with enhancement in open-vocabulary tasks. arXiv e-prints , pages arXiv-2304, 2023.
- [14] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In European Conference on Computer Vision , pages 696-712. Springer, 2022.
- [15] Mengcheng Lan, Chaofeng Chen, Yiping Ke, Xinjiang Wang, Litong Feng, and Wayne Zhang. Clearclip: Decomposing clip representations for dense vision-language inference. In European Conference on Computer Vision , pages 143-160. Springer, 2024.
- [16] Sina Hajimiri, Ismail Ben Ayed, and Jose Dolz. Pay attention to your neighbours: Training-free openvocabulary semantic segmentation. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) , pages 5061-5071. IEEE, 2025.
- [17] Mengcheng Lan, Chaofeng Chen, Yiping Ke, Xinjiang Wang, Litong Feng, and Wayne Zhang. Proxyclip: Proxy attention improves clip for open-vocabulary segmentation. In European Conference on Computer Vision , pages 70-88. Springer, 2024.
- [18] Tong Shao, Zhuotao Tian, Hang Zhao, and Jingyong Su. Explore the potential of clip for training-free open vocabulary semantic segmentation. In European Conference on Computer Vision , pages 139-156. Springer, 2024.

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

- [19] Walid Bousselham, Felix Petersen, Vittorio Ferrari, and Hilde Kuehne. Grounding everything: Emerging localization properties in vision-language transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 3828-3837, 2024.
- [20] Yuhang Yang, Jinhong Deng, Wen Li, and Lixin Duan. Resclip: Residual attention for training-free dense vision-language inference. arXiv preprint arXiv:2411.15851 , 2024.
- [21] Monika Wysocza´ nska, Oriane Siméoni, Michaël Ramamonjisoa, Andrei Bursuc, Tomasz Trzci´ nski, and Patrick Pérez. Clip-dinoiser: Teaching clip a few dino tricks for open-vocabulary semantic segmentation. In European Conference on Computer Vision , pages 320-337. Springer, 2024.
- [22] Dengke Zhang, Fagui Liu, and Quan Tang. Corrclip: Reconstructing correlations in clip with off-the-shelf foundation models for open-vocabulary semantic segmentation. arXiv preprint arXiv:2411.10086 , 2024.
- [23] Barbara Toniella Corradini, Mustafa Shukor, Paul Couairon, Guillaume Couairon, Franco Scarselli, and Matthieu Cord. Freeseg-diff: Training-free open-vocabulary segmentation with diffusion models. ArXiv , abs/2403.20105, 2024.
- [24] Lin Sun, Jiale Cao, Jin Xie, Xiaoheng Jiang, and Yanwei Pang. Cliper: Hierarchically improving spatial representation of clip for open-vocabulary semantic segmentation. arXiv preprint arXiv:2411.13836 , 2024.
- [25] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ArXiv , abs/2010.11929, 2020.
- [26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems , 36:34892-34916, 2023.
- [27] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 26296-26306, 2024.
- [28] Jishnu Mukhoti, Tsung-Yu Lin, Omid Poursaeed, Rui Wang, Ashish Shah, Philip HS Torr, and Ser-Nam Lim. Open vocabulary semantic segmentation with patch aligned contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 19413-19423, 2023.
- [29] Mark Everingham and John Winn. The pascal visual object classes challenge 2012 (voc2012) development kit. Pattern Analysis, Statistical Modelling and Computational Learning, Tech. Rep , 8(5):2-5, 2011.
- [30] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 891-898, 2014.
- [31] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 633-641, 2017.
- [32] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 1209-1218, 2018.
- [33] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. arXiv preprint arXiv:2309.16588 , 2023.
- [34] Patrik O Hoyer. Non-negative matrix factorization with sparseness constraints. Journal of machine learning research , 5(Nov):1457-1469, 2004.
- [35] Zhihui Zhu, Tianyu Ding, Jinxin Zhou, Xiao Li, Chong You, Jeremias Sulam, and Qing Qu. A geometric analysis of neural collapse with unconstrained features. Advances in Neural Information Processing Systems , 34:29820-29834, 2021.
- [36] Jinxin Zhou, Xiao Li, Tianyu Ding, Chong You, Qing Qu, and Zhihui Zhu. On the optimization landscape of neural collapse under mse loss: Global optimality with unconstrained features. In International Conference on Machine Learning , pages 27179-27202. PMLR, 2022.
- [37] Yossi Gandelsman, Alexei A Efros, and Jacob Steinhardt. Interpreting clip's image representation via text-based decomposition. arXiv preprint arXiv:2310.05916 , 2023.

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

- [38] Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. Your large vision-language model only needs a few attention heads for visual grounding. arXiv preprint arXiv:2503.06287 , 2025.
- [39] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread , 1(1):12, 2021.
- [40] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, UweFranke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 3213-3223, 2016.
- [41] MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. https://github.com/open-mmlab/mmsegmentation , 2020.
- [42] Dahyun Kang and Minsu Cho. In defense of lazy visual grounding for open-vocabulary semantic segmentation. In European Conference on Computer Vision , pages 143-164. Springer, 2024.
- [43] Gyungin Shin, Weidi Xie, and Samuel Albanie. Reco: Retrieve and co-segment for zero-shot transfer. Advances in Neural Information Processing Systems , 35:33754-33767, 2022.
- [44] Nikita Araslanov and Stefan Roth. Single-stage semantic segmentation from image labels. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 4253-4262, 2020.
- [45] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning , pages 4904-4916. PMLR, 2021.
- [46] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 2818-2829, 2023.
- [47] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems , 35:25278-25294, 2022.
- [48] Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. International Journal of Computer Vision , 132(2):581-595, 2024.
- [49] Julio Silva-Rodriguez, Sina Hajimiri, Ismail Ben Ayed, and Jose Dolz. A closer look at the few-shot adaptation of large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 23681-23690, 2024.
- [50] Yi-Lin Sung, Jaemin Cho, and Mohit Bansal. Vl-adapter: Parameter-efficient transfer learning for vision-and-language tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 5227-5237, 2022.
- [51] Tao Yu, Zhihe Lu, Xin Jin, Zhibo Chen, and Xinchao Wang. Task residual for tuning vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 10899-10909, 2023.
- [52] Renrui Zhang, Rongyao Fang, Wei Zhang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free clip-adapter for better vision-language modeling. arXiv preprint arXiv:2111.03930 , 2021.
- [53] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 7061-7070, June 2023.
- [54] Huaishao Luo, Junwei Bao, Youzheng Wu, Xiaodong He, and Tianrui Li. Segclip: Patch aggregation with learnable centers for open-vocabulary semantic segmentation. In International Conference on Machine Learning , 2022.

- [55] Jiayun Luo, Siddhesh Khandelwal, Leonid Sigal, and Boyang Albert Li. Emergent open-vocabulary 484 semantic segmentation from off-the-shelf vision-language models. 2024 IEEE/CVF Conference on 485 Computer Vision and Pattern Recognition (CVPR) , pages 4029-4040, 2023. 486

487

488

- [56] Gyungin Shin, Weidi Xie, and Samuel Albanie. Reco: Retrieve and co-segment for zero-shot transfer. ArXiv , abs/2206.07045, 2022.

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

- [57] Jilan Xu, Junlin Hou, Yuejie Zhang, Rui Feng, Yi Wang, Yu Qiao, and Weidi Xie. Learning openvocabulary semantic segmentation models from natural language supervision. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 2935-2944, 2023.
- [58] Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Openvocabulary panoptic segmentation with text-to-image diffusion models. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 2955-2966, 2023.
- [59] Reza Qorbani, Gianluca Villani, Theodoros Panagiotakopoulos, Marc Botet Colomer, Linus HarenstamNielsen, Mattia Segu, Pier Luigi Dovesi, Jussi Karlgren, Daniel Cremers, Federico Tombari, and Matteo Poggi. Semantic library adaptation: Lora retrieval and fusion for open-vocabulary semantic segmentation. ArXiv , abs/2503.21780, 2025.
- [60] Philipp Krähenbühl and Vladlen Koltun. Efficient inference in fully connected crfs with gaussian edge 499 potentials. Advances in neural information processing systems , 24, 2011. 500

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

524

525

526

527

528

529

530

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

543

544

545

546

547

548

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the claims about of analysis discovery and the proposed method, matching our experimental results.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: As noted in the paper, the proposed methods can mitigate but not entirely resolve the decline in spatial discriminability in the final layers.

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

Answer: [NA]

Justification: This paper does not include theoretical results.

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

Justification: We provide detailed descriptions of the experimental settings in Section 4 and the appendix. Additionally, the ablation studies present the rationale behind the choice of hyperparameters used in this work.

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

Answer:[Yes]

Justification: The implementation code is included in the supplementary materials and will be made publicly available.

## Guidelines:

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

Justification: We provide detailed descriptions of the experimental settings in Section 4 and the appendix. Additionally, the ablation studies present the rationale behind the choice of hyperparameters used in this work.

## Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Since our method is training-free and directly uses the pretrained CLIP model weights without any additional optimization, issues related to statistical significance do not arise.

## Guidelines:

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

755

756

757

758

Justification: The details about computer resources used in the experiments are reported in the appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper adheres to the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

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

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: he paper poses no such risks

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The original owners of assets (e.g., code, data, models), used in the paper are properly credited.

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

857

858

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

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

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
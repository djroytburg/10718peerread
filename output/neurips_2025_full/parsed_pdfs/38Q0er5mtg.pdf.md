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

## Equivariant Flow Matching for Point Cloud Assembly

## Anonymous Author(s)

Affiliation Address email

## Abstract

| The goal of point cloud assembly is to reconstruct a complete 3D shape by aligning multiple point cloud pieces. This work presents a novel equivariant solver for assembly tasks based on flow matching models. We first theoretically show that the   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 1 Introduction 11

Point cloud (PC) assembly is a classic machine learning task which seeks to reconstruct 3D shapes 12 by aligning multiple point cloud pieces. This task has been intensively studied for decades and has 13 various applications such as scene reconstruction [48], robotic manipulation [32], cultural relics 14 reassembly [39] and protein designing [41]. A key challenge in this task is to correctly align PC 15 pieces with small or no overlap region, i.e ., when the correspondences between pieces are lacking. 16

17

18

19

20

21

22

23

To address this challenge, some recent methods [32, 40] utilized equivariance priors for pair-wise assembly tasks, i.e ., the assembly of two pieces. In contrast to most of the state-of-the-art methods [30, 51] which align PC pieces based on the inferred correspondence, these equivariant methods are correspondence-free, and they are guided by the equivariance law underlying the assembly task. As a result, these methods are able to assemble PCs without correspondence, and they enjoy high data efficiency and promising accuracy. However, the extension of these works to multi-piece assembly tasks remains largely unexplored.

24

25

26

27

28

In this work, we develop an equivariant method for multi-piece assembly based on flow matching [25].

Our main theoretical finding is that to learn an equivariant distribution via flow matching, one only needs to ensure that the initial noise is invariant and the vector field is related (Thm. 4.2). In other

words, instead of directly handling the

SE

(3)

N

-equivariance for

N

-piece assembly tasks, which can be computationally expensive, we only need to handle the related vector fields on

SE

(3)

N

,

- which is efficient and easy to construct. Based on this result, we present a novel assembly model 29
- called equivariant diffusion assembly (Eda), which uses invariant noise and predicts related vector 30

31

fields by construction. Eda is correspondence-free and is guaranteed to be equivariant by our theory.

- Furthermore, we construct a short and equivariant path for the training of Eda, which guarantees high 32
- data efficiency of the training process. When Eda is trained, an assembly solution can be sampled by 33
- numerical integration, e.g ., the Runge-Kutta method, starting from a random noise. 34
- The contributions of this work are summarized as follows: 35

- -We present an equivariant flow matching framework for multi-piece assembly tasks. Our theory 36 reduces the task of constructing equivariant conditional distributions to the task of constructing 37 related vector fields, thus it provides a feasible way to define equivariant flow matching models. 38
- 39 40 41
- -Based on the theoretical result, we present a simple and efficient multi-piece PC assembly model, called equivariant diffusion assembly (Eda), which is correspondence-free and is guaranteed to be equivariant. We further construct an equivariant path for the training of Eda, which guarantees

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

77

78

79

80

81

82

83

84

high data efficiency.

- -We numerically show that Eda produces highly accurate results on the challenging 3DMatch and BB datasets, and it can even handle non-overlapped pieces.

## 2 Related work

Our proposed method is based on flow matching [25], which is one of the state-of-the-art diffusion models for image generation tasks [11]. Some applications on manifolds have also been investigated [4, 46]. Our model has two distinguishing features compared to the existing methods: it learns conditional distributions instead of marginal distributions, and it explicitly incorporates equivariance priors.

The PC assembly task studied in this work is related to various tasks in literature, such as PC registration [30, 47], robotic manipulation [32, 31] and fragment reassembly [43]. All these tasks aim to align the input PC pieces, but they are different in settings such as the number of pieces, deterministic or probabilistic, and whether the PCs are overlapped. More details can be found in Appx. A. In this work, we consider the most general setting: we aim to align multiple pieces of non-overlapped PCs in a probabilistic way.

Recently, diffusion-based methods have been proposed for assembly tasks, such as registration [6, 18, 44] manipulation [32] and reassembly [34, 45]. However, most of these works simply regard the solution space as a Euclidean space, where the underlying manifold structure and the equivariance priors of the task are ignored. One notable exception is [32], which developed an equivariant diffusion method for robotic manipulation, i.e ., pair-wise assembly tasks. Compared to [32], our method is conceptually simpler because it does not require Brownian diffusion on SO (3) whose kernel is computationally intractable, and it solves the more general multi-piece problem. On the other hand, the invariant flow theory has been studied in [20], which can be regarded as a special case of our theory as discussed in Appx. C.1.

Another branch of related work is equivariant neural networks. Due to their ability to incorporate geometric priors, this type of networks has been widely used for processing 3D graph data such as PCs and molecules. In particular, E3NN [14] is a well-known equivariant network based on the tensor product of the input and the edge feature. An acceleration technique for E3NN was recently proposed [28]. On the other hand, the equivariant attention layer was studied in [12, 22, 24]. Our work is related to this line of approach, because our diffusion network can be seen as an equivariant network with an additional time parameter.

## 3 Preliminaries

This section introduces the major tools used in this work. We first define the equivariances in Sec. 3.1, then we briefly recall the flow matching model in Sec. 3.2.

## 3.1 Equivariances of PC assembly

Consider the action G = ∏ N i =1 SE (3) on a set of N ( N ≥ 2 ) PCs X = { X 1 , . . . , X N } , where SE (3) is the 3D rigid transformation group, ∏ is the direct product, and X i is the i-th PC piece in 3D space. We define the action of g = ( g 1 , . . . , g N ) ∈ G on X as g X = { g i X i } N i =1 , i.e ., each PC X i is rigidly transformed by the corresponding g i . For the rotation subgroup SO (3) N , the action of r = ( r 1 , . . . , r N ) ∈ SO (3) N on X is r X = { r i X i } N i =1 . For SO (3) ⊆ G , we denote r = ( r, . . . , r ) ∈ SO (3) for simplicity, and the action of r on X is written as rX = { rX i } N i =1 .

We also consider the permutation of X . Let S N be the permutation group of N , the action of σ ∈ S N on X is σX = { X σ ( i ) } N i =1 , and the action on g is σ g = ( g σ (1) , . . . , g σ ( N ) ) . For group multiplication,

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

119

120

we denote R ( · ) the right multiplication and L ( · ) the left multiplication, i.e ., ( R r ) r ′ = r ′ r , and ( L r ) r ′ = rr ′ for r , r ′ ∈ SO (3) N .

In our setting, for the given input X , the solution to the assembly task is a conditional distribution P X ∈ µ ( G ) , where µ ( G ) is the set of probability distribution on G . We study the following three equivariances of P X in this work:

Definition 3.1. Let P X ∈ µ ( G ) be a probability distribution on G = SE (3) N conditioned on X , and let ( · ) # be the pushforward of measures.

-P X is SO (3) N -equivariant if ( R r -1 ) # P X = P r X for r ∈ SO (3) N .

-P X is permutation-equivariant if σ # P X = P σX for σ ∈ S N .

-P X is SO (3) -invariant if ( L r ) # P X = P X for r ∈ SO (3) .

Intuitively, the equivariances defined in Def. 3.1 are three natural priors of the assembly task: the SO (3) N -equivariance of P X implies that the solution will be properly transformed when X is rotated; the permutation-equivariance of P X implies that the assembled shape is independent of the order of X ; and the SO (3) -invariance of P X implies that the solution does not have a preferred orientation.

Note that when N = 2 , SO (3) N -equivariance is closely related to SE (3) -bi-equivariance [32, 40], and permutation-equivariance becomes swap-equivariance in [40]. Detailed explanations can be found in Appx. B.

We finally recall the definition of SO (3) -equivariant networks, which will be the main computational tool of this work. We call F l ∈ R 2 l +1 a degreel SO (3) -equivariant feature if the action of r ∈ SO (3) on F l is the matrix-vector production: rF l = R l F l , where R l ∈ R (2 l +1) × (2 l +1) is the degreel Wigner-D matrix of r . We call a network w SO (3) -equivariant if it maintains the equivariance from the input to the output: w ( rX ) = rw ( X ) , where w ( X ) is a SO (3) -equivariant feature. More detailed introduction of equivariances and the underlying representation theory can be found in [3].

## 3.2 Vector fields and flow matching

To sample from a data distribution P 1 ∈ µ ( M ) , where M is a smooth manifold (we only consider M = G in this work), the flow matching [25] approach constructs a time-dependent diffeomorphism φ τ : M → M satisfying ( φ 0 ) # P 0 = P 0 and ( φ 1 ) # P 0 = P 1 , where P 0 ∈ µ ( M ) is a fixed noise distribution, and τ ∈ [0 , 1] is the time parameter. Then the sample of P 1 can be represented as φ 1 ( g ) where g is sampled from P 0 .

Formally, φ τ is defined as a flow, i.e ., an integral curve, generated by a time-dependent vector field v τ : M → TM , where TM is the tangent bundle of M :

<!-- formula-not-decoded -->

According to [25], an efficient way to construct v τ is to define a path h τ connecting P 0 to P 1 . 116 Specifically, let g 0 and g 1 be samples from P 0 and P 1 respectively, and h 0 = g 0 and h 1 = g 1 . v τ 117 can be constructed as the solution to the following problem: 118

<!-- formula-not-decoded -->

When v is learned using (2), we can obtain a sample from P 1 by first sampling a noise g 0 from P 0 and then taking the integral of (1).

In this work, we consider a family of vector fields, flows and paths conditioned on the given PC, 121 and we use the pushforward operator on vector fields to study their relatedness [37]. Formally, 122 let F : M → M be a diffeomorphism, v and w be vector fields on M . w is F -related to v if 123 w ( F ( g )) = F ∗ , g v ( g ) for all g ∈ M , where F ∗ , g is the differential of F at g . Note that we denote 124 v X , φ X and h X the vector field, flow and path conditioned on PC X respectively. 125

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

161

162

163

164

165

166

## 4 Method

In this section, we provide the details of the proposed Eda model. First, the PC assembly problem is formulated in Sec. 4.1. Then, we parametrize related vector fields in Sec. 4.2. The training and sampling procedures are finally described in Sec. 4.3 and Sec. 4.4 respectively.

## 4.1 Problem formulation

Given a set X containing N PC pieces, i.e ., X = { X i } N i =1 where X i is the i -th piece, the goal of assembly is to learn a distribution P X ∈ µ ( G ) , i.e ., for any sample g of P X , g X should be the aligned complete shape. We assume that P X has the following equivariances:

Assumption 4.1. P X is SO (3) N -equivariant, permutation-equivariant and SO (3) -invariant.

We seek to approximate P X using flow matching. To avoid translation ambiguity, we also assume that, without loss of generality, the aligned PCs g X and each input piece X i are centered, i.e ., ∑ i m ( g i X i ) = 0 , and m ( X i ) = 0 for all i , where m ( · ) is the mean vector.

## 4.2 Equivariant flow

The major challenge in our task is to ensure the equivariance of the learned distribution, because a direct implement of flow matching (1) generally does not guarantee any equivariance. To address this challenge, we utilize the following theorem, which claims that when the noise distribution P 0 is invariant and vector fields v X are related, the pushforward distribution ( φ X )# P 0 is guaranteed to be equivariant.

Theorem 4.2. Let G be a smooth manifold, F : G → G be a diffeomorphism, and P ∈ µ ( G ) . If vector field v X ∈ TG is F -related to vector field v Y ∈ TG , then

<!-- formula-not-decoded -->

where P X = ( φ X ) # P 0 , P Y = ( φ Y ) # ( F # P 0 ) . Here φ X , φ Y : G → G are generated by v X and v Y respectively.

Specifically, Thm. 4.2 provides a concrete way to construct equivariant distributions as follow.

Assumption 4.3 (Invariant noise) . P 0 is SO (3) N -invariant, permutation-invariant and SO (3) -invariant, i.e ., ( R r -1 ) # P 0 = P 0 , σ # P 0 = P 0 and P 0 = ( L r ) # P 0 for r ∈ SO (3) N , σ ∈ S N and r ∈ SO (3) .

Corollary 4.4. Under assumption 4.3,

- if v X is R r -1 -related to v r X , then ( R r -1 ) # P X = P r X , where P X = ( φ X ) # P 0 and P r X = ( φ r X ) # P 0 . Here φ X , φ r X : G → G are generated by v X and v r X respectively.
- if v X is σ -related to v σX , then σ # P X = P σX , where P X = ( φ X ) # P 0 and P σX = ( φ σX ) # P 0 . Here φ X , φ σX : G → G are generated by v X and v σX respectively.
- if v X is L r -invariant, i.e., v X is L r -related to v X , then ( L r ) # P X = P X , where P X = ( φ X ) # P 0 .

Nowweconstruct the vector field required by Cor. 4.4. We start by constructing ( R g -1 ) -related vector fields, which are ( R r -1 ) -related by definition, where g ∈ SE (3) N and r ∈ SO (3) N . Specifically, we have the following proposition:

Proposition 4.5. v X is R g -1 -related to v g X if and only if v X ( g ) = ( R g ) ∗ ,e v g X ( e ) for all g ∈ SE (3) N .

According to Prop. 4.5, to construct a ( R g -1 ) -related vector field v X , we only need to parametrize v X at the identity e . Specifically, let f be a neural network parametrizing v X ( e ) , i.e ., f ( X ) = v X ( e ) , we can define v X as

<!-- formula-not-decoded -->

Here, f ( X ) ∈ se (3) N takes the form of

<!-- formula-not-decoded -->

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

The rotation component w i × ( X ) ∈ R 3 × 3 is a skew matrix with elements in the vector w i ( X ) ∈ R 3 , and t i ( X ) ∈ R 3 is the translation component. For simplicity, we omit the superscript i when the context is clear.

Then we enforce the other two relatedness of v X (4). According to the following proposition, σ -relatedness can be guaranteed if f is permutation-equivariant, and L r -invariance can be guaranteed if both w and t are SO(3)-equivariant.

Proposition 4.6. For v X defined in (4),

- if f is permutation-equivariant, i.e ., f ( σX ) = σf ( X ) for σ ∈ S N and PCs X , then σ # v X = v σX ;
- if f is SO(3)-equivariant, i.e ., w ( rX ) = rw ( X ) and t ( rX ) = rt ( X ) for r ∈ SO (3) and PCs X , then ( L r ) # v X = v rX .

Finally, we define P 0 = ( U SO (3) ⊗N (0 , ωI )) N , where U SO (3) is the uniform distribution on SO (3) , N is the normal distribution on R 3 with mean zero and isotropic variance ω ∈ R + , and ⊗ represents the independent coupling. It is straightforward to verify that P 0 indeed satisfies assumption 4.3.

In summary, with P 0 defined above and f (5) satisfying the assumptions in Prop. 4.6, Theorem 4.2 guarantees that the learned distribution has the desired equivariances, i.e ., SO (3) N -equivariance, permutation-equivariance and SO (3) -invariance.

## 4.3 Training

To learn the vector field v X (4) using flow matching (2), we now need to define h X , and the sampling strategy of τ , g 0 and g 1 . A canonical choice [4] is h ( τ ) = g 0 exp( τ log( g -1 0 g 1 )) , where g 0 and g 1 are sampled independently, and τ is sampled from a predefined distribution, e.g ., the uniform distribution U [0 , 1] . However, this definition of h , g 0 and g 1 does not utilize any equivariance property of v X , thus it does not guarantee a high data efficiency.

To address this issue, we construct a 'short' and equivariant h X in the following two steps. First, we independently sample g 0 from P 0 and ˜ g 1 from P X , and obtain g 1 = r ∗ ˜ g 1 , where r ∗ ∈ SO (3) is a rotation correction of ˜ g 1 :

Then, we define h X as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We call h X (7) a path generated by g 0 and ˜ g 1 . Note that h X (7) is a well-defined path connecting g 0 to g 1 , because h X (0) = g 0 and h X (1) = g 1 , and g 1 follows P X (Prop. C.5).

The advantages of h X (7) are twofold. First, instead of connecting a noise g 0 to an independent data sample ˜ g 1 , h X connects g 0 to a modified sample g 1 where the redundant rotation component is removed, thus it is easier to learn. Second, the velocity fields of h X enjoy the same relatedness as v X (4), which leads to high data efficiency. Formally, we have the following observation.

Proposition 4.7 (Data efficiency) . Under assumption 4.3, 4.1, and C.4, we further assume that v X satisfies the relatedness property required in Cor. 4.4, i.e ., v X is R r -1 -related to v r X , v X is σ -related to v σX , and v X is L r -invariant. Denote L ( X ) = E τ, g 0 ∼ P 0 , ˜ g 1 ∼ P X || v X ( h X ( τ )) -∂ ∂τ h X ( τ ) || 2 F the training loss (2) of PC X , where h X is generated by g 0 and ˜ g 1 as defined in (7). Then

<!-- formula-not-decoded -->

- -L ( X ) = L ( σX ) for σ ∈ S N .
- -L ( X ) = ˆ L ( X ) , where ˆ L ( X ) = E τ, g ′ 0 ∼ P 0 , ˜ g ′ 1 ∼ ( L r ) # P X || v X ( h X ( τ )) -∂ ∂τ h X ( τ ) || 2 F is the loss where the data distribution P X is pushed forward by L r ∈ SO (3) .

Prop. 4.7 implies that when h X (7) is combined with the equivariant components developed in Sec. 4.2, the following three data augmentations are not needed: 1) random rotation of each input piece X i , 2) random permutation of the order of the input pieces, and 3) random rotation of the assembled shape, because they have no influence on the training loss.

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

## 4.4 Sampling via the Runge-Kutta method

Finally, when the vector field v X (4) is learned, we can obtain a sample g 1 from P X by numerically integrating v X starting from a noise g 0 from P 0 . In this work, we use the Runge-Kutta (RK) solver on SE (3) N , which is a generalization of the classical RK solver on Euclidean spaces. For clarity, we present the formulations below, and refer the readers to [7] for more details.

To apply the RK method, we first discretize the time interval [0 , 1] into I steps, i.e ., τ i = i I for i = 0 , . . . , I , with a step length η = 1 I . For the given input X , denote f ( g X ) at time τ by f τ ( g ) for simplicity. The first-order RK method (RK1), i.e ., the Euler method, is to iterate:

<!-- formula-not-decoded -->

for i = 0 , . . . , I . To achieve higher accuracy, we can use the fourth-order RK method (RK4):

<!-- formula-not-decoded -->

Note that RK4 (9) is more computationally expensive than RK1 (8), because it requires four evaluations of v X at different points at each step, i.e ., four forward passes of network f , while the Euler method only requires one evaluation per step.

## 5 Implementation

This section provides the details of the network f (5). Our design principle is to imitate the standard transformer structure [38] to retain its best practices. In addition, according to Prop. 4.6, we also require f to be permutationequivariant and SO (3) -equivariant.

The overall structure of the proposed network is shown in Fig. 1. In a forward pass, the input PC pieces { X i } N i =1 are first downsampled using a few downsampling blocks, and then fed into the Croco blocks [42] to model their relations.

Meanwhile, the time step

Figure 1: An overview of our model. The shapes of variables are shown in the brackets.

<!-- image -->

τ

is first embedded using a multi-layer perceptron (MLP) and then incorporated into the above blocks via adaptive normalization [29]. The output is finally obtained by a piece-wise pooling.

Next, we provide details of the equivariant attention layers, which are the major components of both the downsampling block and the Croco block, in Sec. 5.1. Other layers, including the nonlinear and normalization layers, are described in Sec. 5.2.

## 5.1 Equivariant attention layers

Let F l u ∈ R c × (2 l +1) be a channelc degreel feature at point u . The equivariant dot-product attention is defined as:

<!-- formula-not-decoded -->

where 〈· , ·〉 is the dot product, KNN ( u ) ⊆ ⋃ i X i is a subset of points u attends to, K,V ∈ R c × (2 l +1) take the form of the e3nn [14] message passing, and Q ∈ R c × (2 l +1) is obtained by a linear transform:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

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

287

288

289

290

291

292

Here, W l Q ∈ R c × c is a learnable weight, | vu | is the distance between point v and u , ̂ vu = glyph[vector] vu/ | vu | ∈ R 3 is the normalized direction, Y l : R 3 → R 2 l +1 is the degreel spherical harmonic function, c : R + → R is a learnable function that maps | vu | to a coefficient, and ⊗ is the tensor product with the Clebsch-Gordan coefficients.

To accelerate the computation of K and V , we use the SO (2) -reduction technique [28], which rotates the edge uv to the y -axis, so that the computation of spherical harmonic function, the Clebsch-Gordan coefficients, and the iterations of l e are no longer needed. More details are provided in Appx. D.

Following Croco [42], we stack two types of attention layers, i.e ., the self-attention layer and the cross-attention layer, into a Croco block to learn the features of each PC piece while incorporating information from other pieces. For self-attention layers, we set KNN ( u ) to be the k -nearest neighbors of u in the same piece, and for cross-attention layers, we set KNN ( u ) to be the k -nearest neighbors of u in each of the different pieces. In addition, to reduce the computational cost, we use downsampling layers to reduce the number of points before the Croco layers. Each downsampling layer consists of a farthest point sampling (FPS) layer and a self-attention layer.

## 5.2 Adaptive normalization and nonlinear layers

Following the common practice [10], we seek to use the GELU activation function [16] in our transformer structure. However, GELU in its original form is not SO (3) -equivariant. To address this issue, we adopt a projection formulation similar to [9]. Specifically, we define the equivariant GELU (Elu) as:

<!-- formula-not-decoded -->

where ̂ x = x/ ‖ x ‖ is the normalized feature, W ∈ R c × c is a learnable weight. Note that Elu (13) is a natural extension of GELU, because when l = 0 , Elu ( F 0 ) = GELU ( ± F 0 ) .

As for the normalization layers, we use RMS-type layer normalization layers [50] following [23], and we use the adaptive normalization [29] technique to incorporate the time step τ . Specifically, we use the adaptive normalization layer AN defined as:

<!-- formula-not-decoded -->

〈

l

max

1

l

=1

∑

l

l

2

l

+1

F

, F

〉

,

l

is the maximum degree, and to a vector of length

where

σ

=

√

1

c

·

l

max perceptron that maps

τ

max

c

.

We finally remark that the network f defined in this section is SO (3) -equivariant because each layer is SO (3) -equivariant by construction. f is also permutation-equivariant because it does not use any order information of X i .

## 6 Experiment

This section evaluates Eda on practical assembly tasks. After introducing the experiment settings in Sec. 6.1, we first evaluate Eda on the pair-wise registration tasks in Sec. 6.2, and then we consider the multi-piece assembly tasks in Sec. 6.3. An ablation study on the number of PC pieces is finally presented in Sec. 6.4.

## 6.1 Experiment settings

glyph[negationslash]

We evaluate the accuracy of an assembly solution using the averaged pair-wise error. For a predicted assembly g and the ground truth ˆ g , the rotation error ∆ r and the translation error ∆ t are computed as: (∆ r, ∆ t ) = 1 N ( N -1) ∑ i = j ˜ ∆(ˆ g i , ˆ g j g -1 j g i ) , where the pair-wise error ˜ ∆ is computed as ˜ ∆( g, ˆ g ) = ( 180 π accos ( 1 2 ( tr ( r ˆ r T ) -1 )) , ‖ ˆ t -t ‖ ) . Here g = ( r, t ) , ˆ g = (ˆ r, ˆ t ) , and tr ( · ) represents the trace.

For Eda, we use 2 Croco blocks, and 4 downsampling layers with a downsampling ratio 0 . 25 . We use k = 10 nearest neighbors, l max = 2 degree features with d = 64 channels and 4 Following [29], we keep an exponential moving average (EMA) with a decay of 0 . 99 , and we use the AdamW [26] optimizer with a learning rate 10 -4 . Following [11], we use a logit-normal sampling attention heads. for time variable τ . For each experiment, we train Eda on 3 Nvidia A100 GPUs for at most 5 days. We denote Eda with q steps of RK p as 'Eda (RK p , q )' , e.g ., Eda (RK1, 10 ) represents Eda with 10

steps of RK1.

MLP

is a multi-layer

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

## 6.2 Pair-wise registration

This section evaluates Eda on rotated 3DMatch [48] (3DM) dataset containing PC pairs from indoor scenes. Following [17], we consider the 3DLoMatch split (3DL), which contains PC pairs with smaller overlap ratios.

| Table 1: The overlap ratio of PC pairs (%).   | Table 1: The overlap ratio of PC pairs (%).   | Table 1: The overlap ratio of PC pairs (%).   |
|-----------------------------------------------|-----------------------------------------------|-----------------------------------------------|
|                                               | 3DM 3DL                                       | 3DZ                                           |
| Training set                                  | (10 , 100)                                    | 0                                             |
| Test set                                      | (30 , 100) (10 , 30)                          | 0                                             |

Furthermore, to highlight the ability of Eda on non-overlapped assembly tasks, we consider a new split called 3DZeroMatch (3DZ), which contains non-overlapped PC pairs. The comparison of these three splits is shown in Tab. 1.

We compare Eda against the following baseline methods: FGR [52], GEO [30], ROI [47], and AMR [6], where FGR is a classic optimization-based method, GEO and ROI are correspondence-based methods, and AMR is a recently proposed diffusion-like method based on GEO. We report the results of the baseline methods using their official implementations. Note that the correspondence-free methods like [32, 40] do not scale to this dataset.

Table 2: Quantitative results on rotated 3DMatch. ROI (n): ROI with n RANSAC samples.

|               | 3DM   | 3DM   | 3DL   | 3DL   | 3DZ   | 3DZ   |
|---------------|-------|-------|-------|-------|-------|-------|
|               | ∆ r   | ∆ t   | ∆ r   | ∆ t   | ∆ r   | ∆ t   |
| FGR           | 69.5  | 0.6   | 117.3 | 1.3   | -     | -     |
| GEO           | 7.43  | 0.19  | 28.38 | 0.69  | -     | -     |
| ROI (500)     | 5.64  | 0.15  | 21.94 | 0.53  | -     | -     |
| ROI (5000)    | 5.44  | 0.15  | 22.17 | 0.53  | -     | -     |
| AMR           | 5.0   | 0.13  | 20.5  | 0.53  | -     | -     |
| Eda (RK4, 50) | 2.38  | 0.17  | 8.57  | 0.4   | 78.32 | 2.74  |

We report the results in Tab 2. On 3DM and 3DL, we observe that Eda outperforms the baseline methods by a large margin, especially for rotation errors, where Eda achieves more than 50% lower rotation errors on both 3DL and 3DM. We provide more details of Eda on 3DL in Fig. 5 in the appendix.

Figure 2: More details of Eda on 3DZ. A result of Eda is shown in (a) ( ∆ r = 90 . 2 ). Two PC pieces are marked by different colors. ∆ r is centered at 0 , 90 , and 180 on the test set (c), suggesting that Eda learns to keeps the orthogonality or parallelism of walls, floors and ceilings of the indoor scenes.

<!-- image -->

As for 3DZ, we only report the results of Eda in Tab 2, because all baseline methods are not applicable to 3DZ, i.e ., their training goal is undefined when the correspondence does not exist. We observe that Eda's error on 3DZ is much larger compared to that on 3DL, suggesting that there exists much larger ambiguity. We provide an example of the result of Eda in Fig. 2. One important observation is that despite the ambiguity of the data, Eda learned the global geometry of the indoor scenes, in the sense that it tends to place large planes, i.e ., walls, floors and ceilings, in a parallel or orthogonal position.

To show that this behavior is consistent in the whole test set, we present the distribution of ∆ r of Eda on 3DZ in Fig. 2(c). A simple intuition is that for rooms consisting of 6 parallel or orthogonal planes (four walls, a floor and a ceiling), if the orthogonality or parallelism of planes is correctly maintained in the assembly, then ∆ r should be 0 , 90 , or 180 . We observe that this is indeed the case in Fig. 2(c), where ∆ r is centered at 0 , 90 , and 180 . We remark that the ability to learn global geometric properties beyond correspondences is a key advantage of Eda, and it partially explains the superior performance of Eda in Tab. 2

## 6.3 Multi-piece assembly

This section evaluates Eda on the volume constrained version of BB dataset [35]. We consider the shapes with 2 ≤ N ≤ 8 pieces in the 'everyday' subset. We compare Eda against the following baseline methods: DGL [49], LEV [43], GLO [35] and JIG [27]. JIG is correspondence-based, and

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

Figure 4: The results of Eda on different number of pieces.

<!-- image -->

other baseline methods are regression-based. Note that we do not report the results of the diffusiontype method [34] due to accessibility issues. We process all fragments by grid downsampling with a grid size 0 . 02 for Eda. For the baseline methods, we follow their original preprocessing steps. To reproduce the results of the baseline methods, we use the implement of DGL and GLO in the official benchmark suite of BB, and we use the official implement of LEV and JIG.

The results are shown in Tab. 3, where we also report the computation time for the whole test set containing 6904 shapes on a Nvidia T4 GPU. We observe that Eda outperforms all baseline methods by a large margin at a moderate computation cost. We present some qualitative results from Fig. 6 to 8 in the appendix, where we observe that Eda can generally reconstruct the shapes more accurately than the baseline methods. An example of the assembly process of Eda is presented in Fig. 3.

Table 3: Quantitative results on BB dataset and the total computation time on the test set.

|               |    ∆ r |   ∆ t |   Time (min) |
|---------------|--------|-------|--------------|
| GLO           | 126.3  |  0.3  |          0.9 |
| DGL           | 125.8  |  0.3  |          0.9 |
| LEV           | 125.9  |  0.3  |          8.1 |
| JIG           | 106.5  |  0.24 |        122.2 |
| Eda (RK1, 10) |  80.64 |  0.16 |         19.4 |
| Eda (RK4, 10) |  79.2  |  0.16 |         76.9 |

Figure 3: From left to right: the assembly process of a 8 -piece bottle by Eda.

<!-- image -->

## 6.4 Ablation on the number of pieces

This section investigates the influence of the number of pieces on the performance of Eda. We use the kitti odometry dataset [13] containing PCs of city road views. For each sequence of data, we keep pieces that are at least 100 meters apart so that they do not necessarily overlap, and we downsample them using grid downsampling with a grid size 0 . 5 . We train Eda on all consecutive pieces of length 2 ∼ N max in sequences 0 ∼ 8 . We call the trained model Eda-

N max . We then evaluate EdaN max on all consecutive pieces of length M in sequence 9 ∼ 10 .

The results are shown in Fig. 4. We observe that for ∆ r , when the length of the test data is seen in the training set, i.e ., M ≤ N max , Eda performs well, and M &gt; N max leads to worse performance. In addition, Eda-4 generalizes better than Eda-3 on data of unseen length ( 5 and 6 ). The result indicates the necessity of using training data of similar length to the test data. Meanwhile, the translation errors of Eda-4 and Eda-3 are comparable, and they both increase with the length of test data.

## 7 Conclusion and discussion

This work studied the theory of equivariant flow matching, and presented a multi-piece assembly method, called Eda, based on the theory. We show that Eda can accurately assemble PCs on practical datasets.

Eda in its current form has several limitations. First, Eda is slow when using a high order RK solver 372 with a large number of steps. Besides its iterative nature, another cause is the lack of kernel level 373 optimization like FlashAttention [8] for equivariant attention layers. We expect to see acceleration in 374 the future when such optimization is available. Second, Eda always uses all input pieces, which is 375 not suitable for applications like archeology reconstruction, where the input data may contain pieces 376 from unrelated objects. Finally, we have not studied the scaling law [19] of Eda in this work, where 377 we expect to see that an increase in model size leads to an increase in performance similar to image 378 generation applications [29]. 379

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

417

418

419

420

421

422

423

## References

- [1] Federica Arrigoni, Beatrice Rossi, and Andrea Fusiello. Spectral synchronization of multiple views in se (3). SIAM Journal on Imaging Sciences , 9(4):1963-1990, 2016.
- [2] K Somani Arun, Thomas S Huang, and Steven D Blostein. Least-squares fitting of two 3-d point sets. IEEE Transactions on pattern analysis and machine intelligence , (5):698-700, 1987.
- [3] Gabriele Cesa, Leon Lang, and Maurice Weiler. A program to build e (n)-equivariant steerable cnns. In International Conference on Learning Representations , 2022.
- [4] Ricky TQ Chen and Yaron Lipman. Flow matching on general geometries. In The Twelfth International Conference on Learning Representations , 2024.
- [5] Yun-Chun Chen, Haoda Li, Dylan Turpin, Alec Jacobson, and Animesh Garg. Neural shape mating: Self-supervised object assembly with adversarial shape priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 12724-12733, 2022.
- [6] Zhi Chen, Yufan Ren, Tong Zhang, Zheng Dang, Wenbing Tao, Sabine Susstrunk, and Mathieu Salzmann. Adaptive multi-step refinement network for robust point cloud registration. Transactions on Machine Learning Research , 2025.
- [7] Peter E Crouch and R Grossman. Numerical integration of ordinary differential equations on manifolds. Journal of Nonlinear Science , 3:1-33, 1993.
- [8] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS) , 2022.
- [9] Congyue Deng, Or Litany, Yueqi Duan, Adrien Poulenard, Andrea Tagliasacchi, and Leonidas J Guibas. Vector neurons: A general framework for so (3)-equivariant networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 12200-12209, 2021.
- [10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers) , pages 4171-4186, 2019.
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning , 2024.
- [12] Fabian Fuchs, Daniel Worrall, Volker Fischer, and Max Welling. Se(3)-transformers: 3d rototranslation equivariant attention networks. Advances in neural information processing systems , 33:1970-1981, 2020.
- [13] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition , pages 3354-3361. IEEE, 2012.
- [14] Mario Geiger and Tess Smidt. e3nn: Euclidean neural networks. arXiv preprint arXiv:2207.09453 , 2022.
- [15] Zan Gojcic, Caifa Zhou, Jan D Wegner, Leonidas J Guibas, and Tolga Birdal. Learning multiview 3d point cloud registration. In International conference on computer vision and pattern recognition (CVPR) , 2020.
- [16] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415 , 2016.
- [17] Shengyu Huang, Zan Gojcic, Mikhail Usvyatsov, Andreas Wieser, and Konrad Schindler. 424 Predator: Registration of 3d point clouds with low overlap. In Proceedings of the IEEE/CVF 425 Conference on computer vision and pattern recognition , pages 4267-4276, 2021. 426

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

466

467

468

469

470

471

472

- [18] Haobo Jiang, Mathieu Salzmann, Zheng Dang, Jin Xie, and Jian Yang. Se (3) diffusion model-based point cloud registration for robust 6d object pose estimation. Advances in Neural Information Processing Systems , 36:21285-21297, 2023.
- [19] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 , 2020.
- [20] Jonas Köhler, Leon Klein, and Frank Noé. Equivariant flows: exact likelihood generative learning for symmetric densities. In International conference on machine learning , pages 5361-5370. PMLR, 2020.
- [21] Seong Hun Lee and Javier Civera. Hara: A hierarchical approach for robust rotation averaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 15777-15786, 2022.
- [22] Yi-Lun Liao and Tess Smidt. Equiformer: Equivariant graph attention transformer for 3d atomistic graphs. In The Eleventh International Conference on Learning Representations , 2023.
- [23] Yi-Lun Liao, Brandon Wood, Abhishek Das, and Tess Smidt. Equiformerv2: Improved equivariant transformer for scaling to higher-degree representations. arXiv preprint arXiv:2306.12059 , 2023.
- [24] Yi-Lun Liao, Brandon M Wood, Abhishek Das, and Tess Smidt. Equiformerv2: Improved equivariant transformer for scaling to higher-degree representations. In The Twelfth International Conference on Learning Representations , 2024.
- [25] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations , 2023.
- [26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 , 2017.
- [27] Jiaxin Lu, Yifan Sun, and Qixing Huang. Jigsaw: Learning to assemble multiple fractured objects. Advances in Neural Information Processing Systems , 36:14969-14986, 2023.
- [28] Saro Passaro and C Lawrence Zitnick. Reducing so (3) convolutions to so (2) for efficient equivariant gnns. In International Conference on Machine Learning , pages 27420-27438. PMLR, 2023.
- [29] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision , pages 4195-4205, 2023.
- [30] Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing Peng, and Kai Xu. Geometric transformer for fast and robust point cloud registration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 11143-11152, 2022.
- [31] Hyunwoo Ryu, Hong in Lee, Jeong-Hoon Lee, and Jongeun Choi. Equivariant descriptor fields: Se(3)-equivariant energy-based models for end-to-end visual robotic manipulation learning. In The Eleventh International Conference on Learning Representations , 2023.
- [32] Hyunwoo Ryu, Jiwoo Kim, Hyunseok An, Junwoo Chang, Joohwan Seo, Taehan Kim, Yubin Kim, Chaewon Hwang, Jongeun Choi, and Roberto Horowitz. Diffusion-edfs: Bi-equivariant denoising generative modeling on se (3) for visual robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 18007-18018, 2024.
- [33] Hyunwoo Ryu, Hong-in Lee, Jeong-Hoon Lee, and Jongeun Choi. Equivariant descriptor fields: Se (3)-equivariant energy-based models for end-to-end visual robotic manipulation learning. arXiv preprint arXiv:2206.08321 , 2022.

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

- [34] Gianluca Scarpellini, Stefano Fiorini, Francesco Giuliari, Pietro Moreiro, and Alessio Del Bue. Diffassemble: A unified graph-diffusion model for 2d and 3d reassembly. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 28098-28108, 2024.
- [35] Silvia Sellán, Yun-Chun Chen, Ziyi Wu, Animesh Garg, and Alec Jacobson. Breaking bad: A dataset for geometric fracture and reassembly. Advances in Neural Information Processing Systems , 35:38885-38898, 2022.
- [36] Anthony Simeonov, Yilun Du, Andrea Tagliasacchi, Joshua B Tenenbaum, Alberto Rodriguez, Pulkit Agrawal, and Vincent Sitzmann. Neural descriptor fields: Se (3)-equivariant object representations for manipulation. In 2022 International Conference on Robotics and Automation (ICRA) , pages 6394-6400. IEEE, 2022.
- [37] Loring W Tu. Manifolds. In An Introduction to Manifolds , pages 47-83. Springer, 2011.
- [38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems , 30, 2017.
- [39] Haiping Wang, Yufu Zang, Fuxun Liang, Zhen Dong, Hongchao Fan, and Bisheng Yang. A probabilistic method for fractured cultural relics automatic reassembly. Journal on Computing and Cultural Heritage (JOCCH) , 14(1):1-25, 2021.
- [40] Ziming Wang and Rebecka Jörnsten. Se (3)-bi-equivariant transformers for point cloud assembly. In The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS) , 2024.
- [41] Joseph L Watson, David Juergens, Nathaniel R Bennett, Brian L Trippe, Jason Yim, Helen E Eisenach, Woody Ahern, Andrew J Borst, Robert J Ragotte, Lukas F Milles, et al. De novo design of protein structure and function with rfdiffusion. Nature , 620(7976):1089-1100, 2023.
- [42] Philippe Weinzaepfel, Vincent Leroy, Thomas Lucas, Romain Brégier, Yohann Cabon, Vaibhav Arora, Leonid Antsfeld, Boris Chidlovskii, Gabriela Csurka, and Jérôme Revaud. Croco: Self-supervised pre-training for 3d vision tasks by cross-view completion. Advances in Neural Information Processing Systems , 35:3502-3516, 2022.
- [43] Ruihai Wu, Chenrui Tie, Yushi Du, Yan Zhao, and Hao Dong. Leveraging se (3) equivariance for learning 3d geometric shape assembly. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 14311-14320, 2023.
- [44] Yue Wu, Yongzhe Yuan, Xiaolong Fan, Xiaoshui Huang, Maoguo Gong, and Qiguang Miao. Pcrdiffusion: Diffusion probabilistic models for point cloud registration. CoRR , 2023.
- [45] Qun-Ce Xu, Hao-Xiang Chen, Jiacheng Hua, Xiaohua Zhan, Yong-Liang Yang, and Tai-Jiang Mu. Fragmentdiff: A diffusion model for fractured object assembly. In SIGGRAPH Asia 2024 Conference Papers , pages 1-12, 2024.
- [46] Jason Yim, Andrew Campbell, Andrew YK Foong, Michael Gastegger, José Jiménez-Luna, Sarah Lewis, Victor Garcia Satorras, Bastiaan S Veeling, Regina Barzilay, Tommi Jaakkola, et al. Fast protein backbone generation with se (3) flow matching. arXiv preprint arXiv:2310.05297 , 2023.
- [47] Hao Yu, Zheng Qin, Ji Hou, Mahdi Saleh, Dongsheng Li, Benjamin Busam, and Slobodan Ilic. Rotation-invariant transformer for point cloud matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 5384-5393, 2023.
- [48] Andy Zeng, Shuran Song, Matthias Nießner, Matthew Fisher, Jianxiong Xiao, and Thomas Funkhouser. 3dmatch: Learning local geometric descriptors from rgb-d reconstructions. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 18021811, 2017.

- [49] Guanqi Zhan, Qingnan Fan, Kaichun Mo, Lin Shao, Baoquan Chen, Leonidas J Guibas, Hao 520 Dong, et al. Generative 3d part assembly via dynamic graph learning. Advances in Neural 521 Information Processing Systems , 33:6315-6326, 2020. 522
- [50] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural 523 Information Processing Systems , 32, 2019. 524
- [51] Zhengyou Zhang. Iterative point matching for registration of free-form curves and surfaces. 525 International Journal of Computer Vision , 13(2):119-152, 1994. 526
- [52] Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Fast global registration. In Computer Vision527 ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, 528
5. Proceedings, Part II 14 , pages 766-782. Springer, 2016. 529

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

549

550

551

552

553

554

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

565

566

567

## A More details of the related tasks

The registration task aims to reconstruct the scene from multiple overlapped views. A registration method generally consists of two stages: first, each pair of pieces is aligned using a pair-wise method [30], then all pieces are merged into a complete shape using a synchronization method [1, 21, 15]. In contrast to other tasks, the registration task generally assumes that the pieces are overlapped. In other words, it assumes that some points observed in one piece are also observed in the other piece, and the goal is to match the points observed in both pieces, i.e ., corresponding points. The state-ofthe-art registration methods usually infer the correspondences based on the feature similarity [47] learned by neural networks, and then align them using the SVD projection [2] or RANSAC.

The robotic manipulation task aims to move one PC to a certain position relative to another PC. For example, one PC can be a cup, and the other PC can be a table, and the goal is to move the cup on the table. Since the input PCs are sampled from different objects, they are generally non-overlapped. Unlike the other two tasks, this task is generally formulated in a probabilistic setting, as the solution is generally not unique. Various probabilistic models, such as energy based model [36, 31], or diffusion model [32], have been used for this task.

The reassembly task aims to reconstruct the complete object from multiple fragment pieces. This task is similar to the registration task, except that the input PCs are sampled from different fragments, thus they are not necessarily overlapped, e.g ., due to missing pieces or the erosion of the surfaces. Most of the existing methods are based on regression, where the solution is directly predicted from the input PCs [43, 5, 40]. Some probabilistic methods, such as diffusion based methods [45, 34], have also been proposed. Note that there exist some exceptions [27] which assume the overlap of the pieces, and they reply on the inferred correspondences as the registration methods.

A comparison of these three tasks is presented in Tab. 4.

Table 4: Comparison between registration, reassembly and manipulation tasks.

| Task                 | Number of pieces    | Probabilistic/Deterministic   | Overlap        |
|----------------------|---------------------|-------------------------------|----------------|
| Registration         | 2 [30] or more [15] | Deterministic                 | Overlapped     |
| Reassembly           | ≥ 2                 | Deterministic                 | Non-overlapped |
| Manipulation         | 2                   | Probabilistic                 | Non-overlapped |
| Assembly (this work) | ≥ 2                 | Probabilistic                 | Non-overlapped |

## B Connections with bi-equivariance

This section briefly discusses the connections between Def. 3.1 and the equivariances defined in [32] and [40] in pair-wise assembly tasks.

We first recall the definition of the probabilistic bi-equivariance.

Definition B.1 (Eqn. (10) in [32] and Def. (1) in [33]) . ˆ P ∈ µ ( SE (3)) is bi-equivariant if for all g 1 , g 2 ∈ SO (3) , PCs X 1 , X 2 , and measurable set A ⊆ SE (3) ,

<!-- formula-not-decoded -->

Note that we only consider g 1 , g 2 ∈ SO (3) instead of g 1 , g 2 ∈ SE (3) because we require all input PCs, i.e ., X i , g i X i , i = 1 , 2 , to be centered.

Then we recall Def. 3.1 for pair-wise assembly tasks:

Definition B.2 (Restate SO (3) 2 -equivariance and SO (3) -invariance in Def. 3.1 for pair-wise problems) . Let X 1 , X 2 be the input PCs and P ∈ µ ( SE (3) × SE (3)) .

- P is SO (3) 2 -equivariant if P ( A | X 1 , X 2 ) = P ( A ( g -1 1 , g -1 2 ) | g 1 X 1 , g 2 X 2 ) for all g 1 , g 2 ∈ SO (3) and A ⊆ SO (3) × SO (3) , where A ( g -1 1 , g -1 2 ) = { ( a 1 g -1 1 , a 2 g -1 2 ) : ( a 1 , a 2 ) ∈ A } .
- P is SO (3) -invariance if P ( A | X 1 , X 2 ) = P ( rA | X 1 , X 2 ) for all r ∈ SO (3) and A ⊆ SO (3) × SO (3) .

568

569

570

571

572

573

574

575

Intuitively, both Def. B.1 and Def. B.2 describe the equivariance property of an assembly solution, and the only difference is that Def. B.1 describes the special case where X 1 can be rigidly transformed and X 2 is fixed, while Def. B.2 describes the solution where both X 1 and X 2 can be rigidly transformed. In other words, a solution satisfying Def. B.2 can be converted to a solution satisfying Def. B.1 by fixing X 2 . Formally, we have the following proposition.

Proposition B.3. Let P be SO (3) 2 -equivariant and SO (3) -invariant. If ˜ P ( A | X 1 , X 2 ) glyph[defines] P ( A × { e }| X 1 , X 2 ) for A ⊆ SO (3) , then ˜ P is bi-equivariant.

Proof. We prove this proposition by directly verifying the definition.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, the second and the fourth equation hold because P is SO (3) 2 -equivariant, the third equation 576 holds because P is SO (3) -invariant, and the first and last equation are due to the definition. 577

578

579

580

581

582

583

584

585

586

We note that the deterministic definition of bi-equivariance in [40] is a special case of Def. B.1, where ˆ P is a Dirac delta function. In addition, as discussed in Appx. E in [40], a major limitation of the deterministic definition of bi-equivariance is that it cannot handle symmetric shapes. In contrast, it is straightforward to see that the probabilistic definition, i.e ., both Def. B.1 and Def. B.2 are free from this issue. Here, we consider the example in [40]. Assume that X 1 is symmetric, i.e ., there exists g 1 ∈ SO (3) such that g 1 X 1 = X 1 . Under Def. B.1, we have P ( A | X 1 , X 2 ) = P ( A | g 1 X 1 , X 2 ) = P ( Ag 1 | X 1 , X 2 ) , which simply means that P ( A | X 1 , X 2 ) is R g 1 -invariant. Note that this will not cause any contradiction, i.e ., the feasible set is not empty. For example, a uniform distribution on SO (3) is R g 1 -invariant.

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

598

599

600

601

As for the permutation-equivariance, the swap-equivariance in [40] is a deterministic pair-wise version of the permutation-equivariance in Def. B.2, and they both mean that the assembled shape is independent of the order of the input pieces.

## C Proofs

## C.1 Proof in Sec. 4.2

To prove Thm. 4.2, which established the relations between related vector fields and equivariant distributions, we proceed in two steps: first, we prove lemma C.1, which connects related vector fields to equivariant mappings; then we prove lemma. C.2, which connects equivariant mappings to equivariant distributions.

Lemma C.1. Let G be a smooth manifold, F : G → G be a diffeomorphism. If vector field v τ is F -related to vector field w τ for τ ∈ [0 , 1] , then F ◦ φ τ = ψ τ ◦ F , where φ τ and ψ τ are generated by v τ and w τ respectively.

Proof. Let ˜ ψ τ glyph[defines] F ◦ φ τ ◦ F -1 . We only need to show that ˜ ψ τ coincides with ψ τ .

We consider a curve ˜ ψ τ ( F ( g 0 )) , τ ∈ [0 , 1] , for a arbitrary g 0 ∈ G . We first verify that ˜ ψ 0 ( F ( g 0 )) = F ◦ φ 0 ◦ F -1 ◦ F ( g 0 ) = F ( g 0 ) . Note that the second equation holds because φ 0 ( g 0 ) = g 0 , i.e ., φ τ

is an integral path. Then we verify 602

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the 2-nd equation holds due to the chain rule, and the 4 -th equation holds becomes v τ is F -related to w τ . Therefore, we can conclude that ˜ ψ τ ( F ( g 0 )) is an integral curve generated by w τ starting from F ( g 0 ) . However, by definition of ψ τ , ψ τ ( F ( g 0 )) is also the integral curve generated by w τ and starts from F ( g 0 ) . Due to the uniqueness of integral curves, we have ˜ ψ τ = ψ τ .

Lemma C.2. Let φ , ψ , F : G → G be three diffeomorphisms satisfying F ◦ φ = ψ ◦ F . We have F # ( φ # ρ ) = ψ # ( F # ρ ) for all distribution ρ on G .

Proof. Let A ⊆ G be a measurable set. We first verify that φ -1 ( F -1 ( A )) = F -1 ( ψ -1 ( A )) : If x ∈ φ -1 ( F -1 ( A )) , then ( F ◦ φ )( x ) ∈ A . Since F ◦ φ = ψ ◦ F , we have ( ψ ◦ F )( x ) ∈ A , which implies x ∈ F -1 ( ψ -1 ( A )) , i.e ., φ -1 ( F -1 ( A )) ⊆ F -1 ( ψ -1 ( A )) . The other side can be verified similarly. Then we have

<!-- formula-not-decoded -->

which proves the lemma.

Now, we can prove Thm. 4.2 using the above two lemmas.

Proof of Thm. 4.2. Since v X is F -related to v Y , according to lemma C.1, we have F ◦ φ X = φ Y ◦ F . Then according to lemma C.2, we have F # ( φ X # P 0 ) = φ Y # ( F # P 0 ) . The proof is complete by letting P X = φ X # P 0 and P Y = φ Y # ( F # P 0 ) .

Weremark that our theory extends the results in [20], where only invariance is considered, Specifically, we have the following corollary.

Corollary C.3 (Thm 2 in [20]) . Let G be the Euclidean space, F be a diffeomorphism on G , and v τ be a F -invariant vector field, i.e ., v τ is F -related to v τ , then we have F ◦ φ τ = φ τ ◦ F , where φ τ is generated by v τ .

Proof. This is a direct consequence of lemma. C.1 where G is the Euclidean space and w τ = v τ .

Note that the terminology used in [20] is different from ours: The F -invariant vector fields in our work is called F -equivariant vector field in [20], and [20] does not consider general related vector fields.

Finally, we present the proof of Prop. 4.5 and Prop. 4.6.

Proof of Prop. 4.5. If v X is R g -1 -related to v g X , we have v g X (ˆ gg -1 ) = ( R g -1 ) ∗ , ˆ g v X (ˆ g ) for all ˆ g , g ∈ SE (3) N . By letting g = ˆ g , we have

<!-- formula-not-decoded -->

where ( R g ) ∗ ,e = ( ( R g -1 ) ∗ , g ) -1 due to the chain rule of R g R g -1 = e .

On the other hand, if Eqn. (27) holds, we have

<!-- formula-not-decoded -->

which suggests that v X is R g -1 -related to v g X . Note that the second equation holds due to the chain rule of R g -1 R ˆ g = R ˆ gg -1 , and the first and the third equation are the result of Eqn. (27).

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

Proof of Prop. 4.6. 1) Assume v X is σ -related to v σX : ( σ ) ∗ ,g v X ( g ) = V σX ( σ ( g )) . By inserting Eqn. (5) to this equation, we have

<!-- formula-not-decoded -->

Since σ ◦R g = R σ g ◦ σ , by the chain rule, we have σ ∗ ( R g ) ∗ = ( R σ g ) ∗ σ ∗ . In addition, σ ( g ) σ ( X ) = σ ( g X ) . Thus, this equation can be simplified as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first statement in Prop. 4.6 can be proved by reversing the discussion.

2) Assume v X is L r -related to v X : ( L r ) ∗ ,g v X ( g ) = V X ( r g ) . By inserting Eqn. (5) to this equation, we have

<!-- formula-not-decoded -->

Since R r g = R g ◦ R r , by the chain rule, we have ( R r g ) ∗ ,e = ( R g ) ∗ ,r ( R r ) ∗ ,e . In addition, ( L r )( R g ) = ( R g )( L r ) , by the chain rule, we have ( L r ) ∗ , g ( R g ) ∗ ,e = ( R g ) ∗ ,r ( L r ) ∗ ,e . Thus the above equation can be simplified as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By representing f in the matrix form, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all i , where r on the right hand side represents the matrix form of the rotation r . Here the first equation can be equivalently written as w i ( rX ) = rw i ( X ) . The second statement in Prop. 4.6 can be proved by reversing the discussion.

## C.2 Proofs in Sec. 4.3

To establish the results in this section, we need to assume the uniqueness of r ∗ (6):

Assumption C.4. The solution to (6) is unique.

Note that this assumption is mild. A sufficient condition [40] of assumption C.4 is that the singular values of ˜ g T 1 g 0 ∈ R 3 × 3 satisfy σ 1 ≥ σ 2 &gt; σ 3 ≥ 0 , i.e ., σ 2 and σ 3 are not equal. We leave the more general treatment without requiring the uniqueness of r ∗ to future work.

We first justify the definition of g 1 = r ∗ ˜ g 1 by showing that g 1 follows P 1 in the following proposition.

Proposition C.5. Let P 0 and P 1 be two SO (3) -invariant distributions, and g 0 , ˜ g 1 be independent samples from P 0 and P 1 respectively. If r ∗ is given by (6) and assumption C.4 holds, then g 1 = r ∗ ˜ g 1 follows P 1 .

Proof. Define A ˜ g 1 = { g 0 | r ∗ ( g 0 , ˜ g 1 ) = e } , where we write r ∗ as a function of ˜ g 1 and g 0 . Then we have P ( r ∗ = e | ˜ g 1 ) = P 0 ( A ˜ g 1 ) by definition. In addition, due to the uniqueness of the solution to (6), for an arbitrary ˆ r ∈ SO (3) , we have P ( r ∗ = ˆ r | ˜ g 1 ) = P 0 (ˆ rA ˜ g 1 ) . Since P 0 is SO (3) -invariant, we have P 0 (ˆ rA ˜ g 1 ) = P 0 ( A ˜ g 1 ) , thus, P ( r ∗ = ˆ r | ˜ g 1 ) = P ( r ∗ = e | ˜ g 1 ) . In other words, for a given ˜ g 1 , r ∗ follows the uniform distribution U SO (3) .

Finally we compute the probability density of g 1 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which suggests that g 1 follows P 1 . Here the second equation holds because P 1 is SO (3) -invariant. 666 667

which suggests which implies

Then we discuss the equivariance of the constructed h X (7). 668

- Proposition C.6. Given r ∈ SO (3) N , g 0 , ˜ g 1 ∈ SE (3) N , σ ∈ S N , r ∈ SO (3) and τ ∈ [0 , 1] . Let 669 h X be a path generated by g 0 and ˜ g 1 . Under assumption C.4, 670

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

- if h r X is generated by g 0 r -1 and ˜ g 1 r -1 , then h r X ( τ ) = R r -1 h X ( τ ) .
- if h σX is generated by σ ( g 0 ) and σ (˜ g 1 ) , then h σX ( τ ) = σ ( h X ( τ )) .
- if ˆ h X is generated by r g 0 and r ˜ g 1 , then ˆ h X ( τ ) = L r ( h X ( τ )) .

Proof. 1) Due to the uniqueness of the solution to (6), we have r ∗ ( g 0 r -1 , ˜ g 1 r -1 ) = r ∗ ( g 0 , ˜ g 1 ) . Thus, we have

<!-- formula-not-decoded -->

- 2) Due to the uniqueness of the solution to (6), we have r ∗ ( σ ( g 0 ) , σ (˜ g 1 )) = σ ( r ∗ ( g 0 , ˜ g 1 )) . Thus, we have σ ( h X ) = h σX .
- 3) Due to the uniqueness of the solution to (6), we have r ∗ ( r g 0 , r ˜ g 1 ) = rr ∗ ( g 0 , ˜ g 1 ) r -1 . Thus,

<!-- formula-not-decoded -->

With the above preparation, we can finally prove Prop. 4.7.

Proof of Prop. 4.7. 1) By definition

<!-- formula-not-decoded -->

where h r X is the path generated by g ′ 0 and ˜ g ′ 1 . Since P 0 = ( R r -1 ) # P 0 and P r X = ( R r -1 ) # P X by 682 assumption, we can write g ′ 0 = g 0 r -1 and ˜ g ′ 1 = ˜ g 1 r -1 , where g 0 ∼ P 0 and ˜ g 1 ∼ P X . According to 683 the first part of Prop. C.6, we have h r X ( τ ) = R r -1 h X ( τ ) , where h X is a path generated by g 0 and ˜ g 1 . 684 By taking derivative on both sides of the equation, we have ∂ ∂τ h r X ( τ ) = ( R r -1 ) ∗ ,h X ( τ ) ∂ ∂τ h X ( τ ) . 685 Then we have 686

<!-- formula-not-decoded -->

by inserting these two equations into Eqn. (42). Since v X is R r -1 -related to v r X by assumption, we 687 have v r X ( R r -1 h X ( τ )) = ( R r -1 ) ∗ ,h X ( τ ) v X ( h X ( τ )) . Thus, we have 688

<!-- formula-not-decoded -->

689

690

691

692

693

694

695

696

697

where the second equation holds because ( R r -1 ) ∗ ,h X ( τ ) is an orthogonal matrix. The desired result follows.

2) The second statement can be proved similarly as the first one, where σ -equivariance is considered instead of R r -1 -equivariance.

3) Denote g ′ 0 = r g 0 and ˜ g ′ 1 = r ˜ g 1 , where g 0 ∼ P 0 and ˜ g 1 ∼ P X . According to the third part of Prop. C.6, we have ˆ h X ( τ ) = L r ( h X ( τ )) . By taking derivative on both sides of the equation, we have ∂ ∂τ ˆ h X ( τ ) = ( L r ) ∗ ,h X ( τ ) ∂ ∂τ h X ( τ ) . Then the rest of the proof can be conducted similarly to the first part of the proof.

## D SO (2) -reduction

The main idea of SO (2) -reduction [] is to rotate the edge uv to the y -axis, and then update node 698 feature in the rotated space. Since all 3D rotations are reduced to 2D rotations about the y -axis in the 699 rotated space, the feature update rule is greatly simplified. 700

701

702

703

704

705

706

707

708

Here, we describe this technique in the matrix form to facilitates better parallelization. The original element form description can be found in []. Let F l v ∈ R c × (2 l +1) be a c -channel l -degree feature of point v , and L &gt; 0 be the maximum degree of features. We construct ˆ F l v ∈ R c × (2 L +1) by padding F l v with L -l zeros at the beginning and the end of the feature, then we define the full feature F v ∈ R c × L × (2 L +1) as the concatenate of all ˆ F l v with 0 &lt; l ≤ L . For an edge vu , there exists a rotation r vu that aligns uv to the y -axis. We define R vu ∈ R L × (2 L +1) × (2 L +1) to be the full rotation matrix, where the l -th slice R vu [ l, : , :] is the l -th Wigner-D matrix of r vu with zeros padded at the boundary. K v defined in (11) can be efficiently computed as

<!-- formula-not-decoded -->

where M 1 × i M 2 represents the batch-wise multiplication of M 1 and M 2 with the i -th dimen709 sion of M 2 treated as the batch dimension. W K ∈ R ( cL ) × ( cL ) is a learnable weight, D K ∈ 710 R c × (2 L +1) × (2 L +1) is a learnable matrix taking the form of 2D rotations about the y -axis, i.e ., for 711 each i , D K [ i, : , :] is 712

<!-- formula-not-decoded -->

where a 1 , · · · , a L , b 1 , · · · , b L -1 : R + → R are learnable functions that map | vu | to the coefficients. 713 V v defined in (11) can be computed similarly. Note that (45) does not require the computation of 714 Clebsch-Gordan coefficients, the spherical harmonic functions, and all computations are in the matrix 715 form where no for-loop is needed, so it is much faster than the computations in (11). 716

717

## E More details of Sec. 6

We present more details of Eda on 3DL in Fig. 5. We observe that the vector field is is gradually 718 learned during training, i.e ., the training error converges. On the test set, RK4 outperforms the RK1, 719 and they both benefit from more time steps, especially for rotation errors. 720

Figure 5: More details of Eda on 3DL. Left: the training curve. Middle and right: the influence of RK4/RK1 and the number of time steps on ∆ r and ∆ t .

<!-- image -->

We provide the complete version of Table 2 in Table 5, where we additionally report the standard 721 deviations of Eda. 722

We provide some qualitative results on BB datasets in Fig. 6 and Fig. 8. Eda can generally recover the 723 shape of the objects except for some rare cases, such as the 3 rd sample in the second row in Fig. 6. 724 We hypothesize that Eda can achieve better performance when using finer grained inputs. A complete 725 version of Tab. 3 is provided in Tab. 6, where we additionally report the standard deviations of Eda. 726

We provide a few examples of the reconstructed road views in Fig. 9. 727

Table 5: The complete version of Table 2 with stds of Eda reported in bracked.

|               | 3DM         | 3DM         | 3DL         | 3DL       | 3DZ         | 3DZ         |
|---------------|-------------|-------------|-------------|-----------|-------------|-------------|
|               | ∆ r         | ∆ t         | ∆ r         | ∆ t       | ∆ r         | ∆ t         |
| FGR           | 69.5        | 0.6         | 117.3       | 1.3       | -           | -           |
| GEO           | 7.43        | 0.19        | 28.38       | 0.69      | -           | -           |
| ROI (500)     | 5.64        | 0.15        | 21.94       | 0.53      | -           | -           |
| ROI (5000)    | 5.44        | 0.15        | 22.17       | 0.53      | -           | -           |
| AMR           | 5.0         | 0.13        | 20.5        | 0.53      | -           | -           |
| Eda (RK4, 50) | 2.38 (0.16) | 0.16 (0.01) | 8.57 (0.08) | 0.4 (0.0) | 78.74 (0.6) | 0.96 (0.01) |

Table 6: The complete version of Table 3 with stds of Eda reported in brackets.

|               | ∆ r         | ∆ t        |   Time (min) |
|---------------|-------------|------------|--------------|
| GLO           | 126.3       | 0.3        |          0.9 |
| DGL           | 125.8       | 0.3        |          0.9 |
| LEV           | 125.9       | 0.3        |          8.1 |
| Eda (RK1, 10) | 80.64       | 0.16       |         19.4 |
| Eda (RK4, 10) | 79.2 (0.58) | 0.16 (0.0) |         76.9 |

Figure 6: Qualitative results of Eda and DGL.

<!-- image -->

Figure 7: Qualitative results of GLO and LEV.

<!-- image -->

Figure 9: Qualitative results of Eda on kitti. We present the results of Eda (1-st row) and the ground truth (2-nd row). For each assembly, Eda correctly places the input road views on the same plane.

<!-- image -->

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

773

774

775

776

777

778

779

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We claim a new equivariant flow matching method, called Eda, for multi-piece assembly tasks in the abstract and introduction. This is the main contribution of the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations/future research directions in Sec. 7.

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

Justification: All proofs can be found in Sec. C

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

Justification: The details of the experiments are provided in Sec. 6.1,

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

Justification: We will release the code needed to reproduce the results upon acceptance.

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

Justification: The details of the experiments are provided Sec. 6.1,

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We provide standard deviation for our method in Tab. 5 and Tab. 6. We also include std in the figures in Sec. 6.4.

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

Justification: As stated in Sec. 6.1, all experiments are conducted on a cluster with 3 A100 GPUs.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in the paper conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This work focuses on point cloud assembly task. The author does not see any negative societal impact of this work.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: This work does not generate new data. The author does not see any risk of misuse.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: This work used 3DMatch, BB and kitti datasets and cited them. The baseline methods are also cited.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

| 1041      | 16.   | Declaration of LLMusage                                                                                                                                                                                                                                                                                                                                             |
|-----------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1042      |       | Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. |
| 1043      |       | Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. |
| 1044      |       | Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. |
| 1045      |       | Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. |
| 1046      |       | Answer: [NA]                                                                                                                                                                                                                                                                                                                                                        |
| 1047      |       | Justification: The core method development in this research does not involve LLMs.                                                                                                                                                                                                                                                                                  |
| 1048      |       | Guidelines:                                                                                                                                                                                                                                                                                                                                                         |
| 1049 1050 |       | • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.                                                                                                                                                                                                               |
| 1051      |       | • Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.                                                                                                                                                                                                                                            |
| 1052      |       | • Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.                                                                                                                                                                                                                                            |
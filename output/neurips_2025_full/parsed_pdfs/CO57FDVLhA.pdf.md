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

## Quantization vs Pruning: Insights from the Strong Lottery Ticket Hypothesis

## Anonymous Author(s)

Affiliation Address email

## Abstract

Quantization is an essential technique for making neural networks more efficient, yet our theoretical understanding of it remains limited. Previous works demonstrated that extremely low-precision networks, such as binary networks, can be constructed by pruning large, randomly-initialized networks, and showed that the ratio between the size of the original and the pruned networks is at most polylogarithmic.

The specific pruning method they employed inspired a line of theoretical work known as the Strong Lottery Ticket Hypothesis (SLTH), which leverages insights from the Random Subset Sum Problem. However, these results primarily address the continuous setting and cannot be applied to extend SLTH results to the quantized setting.

In this work, we build on foundational results by Borgs et al. on the Number Partitioning Problem to derive new theoretical results for the Random Subset Sum Problem in a quantized setting. Using these results, we then extend the SLTH framework to finite-precision networks. While prior work on SLTH showed that pruning allows approximation of a certain class of neural networks, we demonstrate that, in the quantized setting, the analogous class of target discrete neural networks can be represented exactly, and we prove optimal bounds on the necessary overparameterization of the initial network as a function of the precision of the target network.

## 1 Introduction

Deep neural networks (DNNs) have become ubiquitous in modern machine-learning systems, yet their ever-growing size quickly collides with the energy, memory, and latency constraints of real-world hardware.

Quantization

-representing weights with a small number of bits-is arguably the most hardware-friendly compression technique, and recent empirical work shows that aggressive

quantization can preserve accuracy even down to the few bits regime. Unfortunately, our theoretical understanding of why and when such extreme precision reduction is possible still lags far behind

practice. An interesting step in this direction was the

Multi-prize Lottery Ticket Hypothesis put forward by Diffenderfer and Kailkhura [2021]. They

large, randomly initialized network contains sparse empirically

(MPLTH)

demonstrated that a sufficiently binary

subnetworks that match the performance of a target network with real-valued weights. They also provided theoretical guarantees regarding

- the existence of such highly quantized networks, showing that, with respect to the target network, 32
- the initial random network need only be larger by a polynomial factor . Sreenivasan et al. [2022] 33
- subsequently improved this bound, by showing that a polylogarithmic factor is sufficient (See Section 34
- 2). These works fall within the research topic known as the Strong Lottery Ticket Hypothesis (SLTH), 35
- which states that sufficiently-large randomly initialized neural networks contain subnetworks, called 36

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

77

78

79

80

81

82

83

84

85

lottery tickets , that perform well on a given task, without requiring weight adjustments. The main theoretical question, therefore, is: how large should the initial network be to ensure it contains a lottery ticket capable of approximating a given family of target neural networks? Research on the SLTH, however, has mainly focused on investigating pruning in the continuous-weight (i.e. not quantized) setting, drawing on results for the Random Subset Sum Problem (RSSP) [Lueker, 1998] to show that over-parameterized networks can be pruned to approximate any target network without further training [Orseau et al., 2020, Pensia et al., 2020, Burkholz, 2022a] (for additional context on this body of literature, we kindly refer the reader to the Related Work in Section 2). However, the analytic RSSP results used for SLTH rely heavily on real-valued weights and therefore do not extend to the finite-precision regime considered in the MPLTH. This gap left open a fundamental question:

What is the over-parameterization needed to obtain quantized strong lottery tickets?

Our contributions. We address the aforementioned gap by revisiting the classic Number Partitioning Problem (NPP), which is closely related to the RSSP. Building on the seminal results of Borgs et al. [2001] concerning the phase transition of NPP, we derive new, sharp bounds for the discrete RSSP . These bounds are precise enough to adapt the SLTH proof strategy to the finite precision setting and, in doing so, establish optimal bounds for the MPLTH. Crucially, our results account for arbitrary quantization in both the initial and target networks, and demonstrate that the lottery ticket can represent the target network exactly . In contrast, prior work limited the initial network to binary weights and assumed continuous weights for the larger (target) network [Diffenderfer and Kailkhura, 2021, Sreenivasan et al., 2022], requiring a cubic overparameterization relative to the lower bound and additional dependencies on network parameters absent in our bound. Concretely, let δ t denote the precision (i.e., quantization level) of a target network N t , δ in the precision of a randomly initialized larger network N in, and δ any parameter satisfying δ t ≥ δ 2 ≥ δ 2 in . Denote by d and ℓ the width and depth of the target network, respectively. Our results can be summarized by the following simplified, informal theorem (refer to the formal statements for full generality).

Theorem (Informal version of Theorems 1, 2, and 3) . With high probability 1 , a depth2 ℓ network N in of width O ( d log(1 /δ ) ) can be quantized to precision δ and pruned to become functionally equivalent to any δ t -quantized target network N t with layers of width at most d (Theorem 1). This result is optimal, as no two-layer network of precision δ with fewer than Ω ( d log(1 /δ ) ) parameters can be pruned to represent δ -quantized neural networks of width d (Theorem 3). Furthermore, the depth of N in can be reduced to ( ℓ +1) at the cost of an additional log(1 /δ ) factor in its width (Theorem 2).

These are the first theoretical results that (i) characterize the precise interplay between weight precision and over-parameterization, and (ii) certify that pruning can yield exact , not just approximate, quantized subnetworks. Besides contributing to the theory of network compression, our analysis showcases the versatility of classical combinatorial insights-such as the theory of NPP-in deep-learning theory.

Paper organization. In Section 2, we review prior work on SLTH and quantization. In Section 3, we prove a new quantized version of the RSSP, after first recalling classical results on RNPP in subsection 3.1. Our new theorems on the quantized SLTH are proved in Sections 4, after recalling necessary notation and definitions in subsection 4.1. Finally, in Section 5, we discuss the conclusion of our work and future directions.

## 2 Related Work

Strong Lottery Ticket Hypothesis. In 2018, Frankle and Carbin [2019] proposed the Lottery Ticket Hypothesis, which states that every dense network contains a sparse subnetwork that can be trained from scratch, and performs equally well as the dense network. Rather surprisingly, Zhou et al. [2019], Ramanujan et al. [2020] and Wang et al. [2019] empirically showed that it is possible to efficiently find subnetworks within large randomly initialized networks that perform well on a given task, without changing the initial weights. This motivated the Strong Lottery Ticket Hypothesis (SLTH), which states that sufficiently overparameterized randomly initialized neural networks contain sparse subnetworks that will perform as well as a small trained network on a given dataset, without

1 As customary in the literature on randomized algorithms, with the expression with high probability we refer to a probability of failure which scales as the inverse of a polynomial in the parameter of interest (the number of precision bits log 1 /δ ) in our case).

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

any training. Many formal results rigorously proved the SLTH in various settings, the first one being Malach et al. [2020], where they showed that a feed-forward dense target network of width ℓ and depth d can be approximated by pruning a random network of depth 2 ℓ and width O ( d 5 ℓ 2 ) . Orseau et al. [2020], Pensia et al. [2020] improved this bound by proving that width O ( d log( dℓ )) is sufficient. Another construction was provided by Burkholz [2022a], where they showed that a network of width ℓ +1 is enough to approximate a network of width ℓ , with a certain compromise on the width. Other works extended the SLTH to other famous architectures, such as convolutional Burkholz [2022b] and equivariant networks Ferbach et al. [2022]. Next, we provide an informal version that qualitatively summarizes this kind of results.

Theorem (Informal qualitative template of SLTH results) . With high probability, a random artificial neural network N R with m parameters can be pruned so that the resulting subnetwork approximates, up to an error ϵ , any target artificial neural network N T with O ( m/ log 2 (1 /ϵ )) parameters. The logarithmic dependency on ϵ is optimal.

Quantization. Neural network quantization refers to the process of reducing the precision of the weights within a neural network. Empirical studies have demonstrated that trained neural networks can often be significantly quantized without incurring substantial loss in performance Han et al. [2015]. In particular, Diffenderfer and Kailkhura [2021] provided both empirical and theoretical support for a quantized variant of the SLTH, introducing an algorithm capable of training binary networks effectively. With regard to theoretical guarantees, they proved that a neural network with width d and depth ℓ can be approximated to within an error ϵ by a binary target network of width O ( ℓd 3 / 2 /ϵ + ℓd log( ℓd/ϵ )) . Subsequently, Sreenivasan et al. [2022] presented an exponential improvement over this result, demonstrating that a binary network with depth Θ( ℓ log( dℓ/ϵ )) and width Θ( d log 2 ( dℓ/ϵ )) suffices to approximate any given network of width d and depth ℓ . We remark that both of these results assume that the initial network weights are binary, whereas the target network weights are continuous. The success of techniques to construct heavily quantized networks can be related to theoretical work that show that heavily quantized networks still retain good universal approximation properties [Hwang et al., 2024]. In practice, not all parts of a network need to be quantized equally aggressively. Mixed-precision quantization allocates different bit-widths to different layers or parameters to balance accuracy and efficiency [Carilli, 2020, Younes Belkada, 2022].

Subset Sum Problem (SSP). Given a target value z and a multiset Ω of n integers from the set {-M, -M +1 , . . . , M -1 , M } , the SSP consists in finding a subset S ⊆ Ω such that the sum of its elements equal z . In Computational Complexity Theory, SSP is one of the most famous NP-complete problems Garey and Johnson [1979]. Its random version, Random SSP (RSSP), has been investigated since the 80s in the context of combinatorial optimization [Lueker, 1982, 1998], and recently received renowned attention in the machine learning community because of its connection to the SLTH [Pensia et al., 2020].

Number Partitioning Problem (NPP). NPP is the problem of partitioning a multiset Ω of n integers from the set [ M ] := { 1 , 2 , . . . , M } into two subsets such that the difference of their respective sums equals a target value z (typically, the literature has focused on minimizing | z | , i.e. trying to approximate the value closest to zero). Analogously to the aforementioned SSP, NPP is one of the most important NP-complete problems [Garey and Johnson, 1979, Hayes, 2002]. Its random version, in which the n elements are sampled uniformly at random from [ M ] , has also received considerable attention in Statistical Physics, where it has been shown to exhibit a phase transition Mézard and Montanari [2009]. Concretely, Mertens [1998] heuristically showed the following result, which was later put on rigorous grounds by Borgs et al. [2001]: defining κ := log 2 M n , if κ &lt; 1 then O (2 n ) number of solutions exist, whereas if κ &gt; 1 the number of solutions sharply drops to zero.

## 3 Quantized Random Subset Sum Problem

## 3.1 Random Number Partitioning Problem

In this section, we recall seminal results by Borgs et al. [2001] which we leverage in Subsection 3.2 to obtain new results on the quantized RSSP.

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

Definition 1 (RNNP) . Let X = ( X 1 , X 2 , . . . , X n ) be a set of integers sampled uniformly from the set { 1 , 2 , . . . , M } . The Random Number Partitioning Problem is defined as the problem of finding a partitioning set σ = ( σ 1 , σ 2 , . . . , σ n ) with σ j ∈ {-1 , 1 } such that | σ · x | = z for some given integer z (called target).

Note that usually, in RNPP the difference between the sum of two parts is minimized, but we consider RNPP with a target, i.e., the difference between the sum of two parts must be equal to a given number z , the target. Given an instance of Random Number Partitioning Problem X = ( X 1 , X 2 , . . . , X n ) with a set of size n and a target z , Z n,z denotes the number of exact solutions to the RNPP, i.e.,

<!-- formula-not-decoded -->

To prove the existence of phase transition, Borgs et al. [2001] estimated the moments of Z n,z . The relevant result is stated as Theorem 4 (Appendix A). Using these moment estimates of Z n,z , one can write an upper and a lower bound on the probabilities of existence of solutions to a RNPP. We do so in Lemma 1.

Lemma 1. Given a Random Number Partitioning Problem, the probability P ( Z n,z &gt; 0) is bounded above and below as

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ρ n is defined as ρ n = 2 n +1 γ n .

Proof Sketch. We use Markov's Inequality (Theorem 5 in Appendix C) and Cauchy-Schwartz inequality (Theorem 6 in Appendix C) to get bounds on the probabilities from the moment estimates (Theorem 4). See Appendix A for details.

The existence of phase transition (Section 2) is a consequence of Lemma 1 but for the purposes of this paper, we only require Lemma 1.

## 3.2 Quantized Random Subset Sum Problem

The Random Subset Sum Problem (RSSP) is the problem of finding a subset of a given set such that the sum of this subset equals a given target t . RSSP is a crucial tool in proving results on SLTH Pensia et al. [2020] Burkholz [2022a]. RSSP and RNPP are closely related, and hence we can use the results on RNPP in this section to make statements about RSSP. We shall then use these results on RSSP to prove results on SLTH and quantization.

Definition 2 (RSSP) . Let X = ( X 1 , X 2 , . . . , X n ) be a set of integers sampled uniformly from the set {-M,..., 1 , 2 , 3 , . . . , M } . The RSSP is defined as the problem of finding an index set S ⊂ [ n ] such that ∑ i ∈ S X i = t for a given integer t , called the target.

Lemma 2. An SSP with given set X = ( X 1 , X 2 , . . . , , X n ) and target t can be solved iff the NPP can be solved on the given set X and target Λ -2 t (or 2 t -Λ ), where Λ = ∑ n i =1 X i .

Lemma 2 is proved in Appendix A. Using the equivalence of RNPP and RSSP, the following results on RSSP follows from Lemma 1.

Lemma 3. Consider a RSSP on the set X = ( X 1 , X 2 , . . . , X n ) where X i 's are sampled uniformly from {-M,..., -1 , 1 , . . . , M } with a target t = O ( M ) . Let Y n,t be the number of possible solutions to the RSSP problem. Then

̸

<!-- formula-not-decoded -->

174

where 175

<!-- formula-not-decoded -->

Proof Sketch. We first convert the given RSSP to a RNPP through the transformation in Lemma 2. 176 The result then follows from Lemma 1. See Appendix A for details. 177

178

The next lemma shows under what condition an RSSP can be solved with high probability.

179

180

then we have 181

Lemma 4. Let M = M ( n ) be an arbitrary function of n . Consider as RSSP on the set X = ( X 1 , X 2 , . . . , X n ) sampled uniformly from {-M,..., -1 , 1 , . . . , M } with a target t = O ( M ) . If

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof Sketch. It can be shown using Hoeffding's inequality (Theorem 7 in Appendix C), that with 182 high probability the sum of all elements satisfies Λ &lt; √ 2 7 M √ n log n . Hence, the probability of 183 solving a RSSP from Lemma 3 can be analyzed under the assumption of κ n &lt; 1 . See Appendix A 184 for details. 185

186

187

## 4 SLTH and Weight Quantization

## 4.1 Notation and Setup

In this Subsection, we define some notation before stating our results. Scalars are denoted by 188 lowercase letters such as w , y , etc. Vectors are represented by bold lowercase letters, e.g., v , and 189 the i th component of a vector v is denoted by v i . Matrices are denoted by bold uppercase letters 190 such as M . If a matrix W has dimensions d 1 × d 2 , we write W ∈ R d 1 × d 2 . We define the finite set 191 S δ := {-1 , -1 + δ, -1 + 2 δ, . . . , 1 } , where δ = 2 -k for some k ∈ N . A real number b is said to 192 have precision δ if b ∈ S δ . We denote the d -fold Cartesian product of S δ by S d δ ; that is, 193

<!-- formula-not-decoded -->

For w ∈ S δ with δ = 2 -k and γ = 2 -m such that k &gt; m , we define the quantization operator [ · ] γ by 194

<!-- formula-not-decoded -->

This operation reduces the precision of w to γ . For a vector v , the notation w = [ v ] γ means w i = [ v i ] γ for all components i . We use C, C i for i ∈ N to denote positive absolute constants.

Definition 3. An ℓ -layer neural network is a function f : R d 0 → R d ℓ defined as

<!-- formula-not-decoded -->

where W i ∈ R d i × d i -1 for i = 1 , . . . , ℓ , x ∈ R d 0 , and σ : R → R is a nonlinear activation function. 198 For a vector x , the expression v = σ ( x ) denotes componentwise application: v i = σ ( x i ) . 199

195

196

197

<!-- formula-not-decoded -->

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

The entries of the matrices W i are referred to as the weights or parameters of the network. In this work, we assume all activation functions are ReLU, i.e., σ ( x ) = max(0 , x ) . This assumption is made for simplicity; the results can be extended to general activation functions as discussed in Burkholz [2022a].

For a neural network f ( x ) = W ℓ σ ( W ℓ -1 · · · σ ( W 1 x )) , we refer to the quantity σ ( W k · · · σ ( W 1 x )) as the output of the k th layer.

We next define some quantization strategies for neural networks which capture mixed-precision quantization practices. We defer the reader to the quantization paragraph in the Related Work (Section 2) for a discussion of such practices.

Definition 4. A δ -quantized neural network is a neural network whose weights are sampled uniformly from the set S δ = {-1 , . . . , δ, . . . , 1 } , where δ = 2 -k for some k ∈ N .

Definition 5. A neural network f is called a γ -double mixed precision neural network if the output of each layer is quantized to precision γ , i.e.,

<!-- formula-not-decoded -->

Definition 6. A neural network f is called an γ -triple mixed precision neural network if the outputs of its even-numbered layers are quantized to precision γ , i.e.,

<!-- formula-not-decoded -->

More generally, a mixed-precision neural network may reset the precision to γ at some layers while leaving others unquantized. Reducing the precision of a δ -quantized neural network f to γ means all weights of f are mapped to [ · ] γ . We denote this operation as [ f ] γ .

Our objective is to represent a target Double Mixed Precision neural network f , with weights which are δ 1 quantized, using a second, potentially overparameterized, mixed-precision network g with finer quantization δ 2 , by quantizating and pruning it. For a neural network

<!-- formula-not-decoded -->

the pruned network g S i is defined as:

<!-- formula-not-decoded -->

where each S i is a binary pruning mask with the same dimensions as M i , and ⊙ denotes elementwise multiplication. The goal is to find masks S 1 , S 2 , . . . , S ℓ such that f can be represented by the quantized and pruned version of g .

## 4.2 Quantized SLTH Results

Having discussed the pervious work on NPP and it's connection to RSSP, we now apply these results to prove results on SLTH in quantized setting. The main question that we want to answer is the following: Suppose we are given a target neural network, whose weights are of precision δ t and a large network whose weights are of precision δ in, such that δ t ≥ δ in. Suppose we have the freedom to reduce the precision of the large network to δ , and then we can prune it. What is the relationship between δ and size of the large network such that the bigger network can be pruned to the target network. Now we state our first main result, which is analogs to the theorem proved by Pensia et al. [2020], but in the quantized setting.

Theorem 1. Let F be the class of δ t quantized γ -double mixed Precision neural networks of the form

<!-- formula-not-decoded -->

Consider a 2 ℓ layered randomly initialized δ in-quantized γ -touble mixed Precision neural network

<!-- formula-not-decoded -->

with δ 2 in ≤ δ t . Let δ 2 in ≤ δ 2 ≤ δ t . Assume M 2 i has dimension

<!-- formula-not-decoded -->

and M 2 i -1 has dimension 239

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

265

266

267

268

with probability at least we have ∀ w ∈ S δ t

<!-- formula-not-decoded -->

Proof. Let the precision of g be δ . First decompose wx = σ ( wx ) -σ ( -wx ) . This is a general identity for ReLU non-linear activation and was introduced in Malach et al. [2020]. WLOG 2 say w &gt; 0 . Let

<!-- formula-not-decoded -->

where a , b , c , d ∈ R n , s 1 1 , s 1 2 , s 2 1 , s 2 2 ∈ { 0 , 1 } n . This is shown diagrammatically in Figure 1 in Appendix D.

Step 1: Let a + = max { 0 , a } be the vector obtained by pruning all the negative entries of a . This is done by appropriately choosing s 1 1 Since w ≥ 0 , then for all x ≤ 0 we have σ ( wx ) = b T σ ( a + x ) = 0 . Moreover, further pruning of a + would not affect this equality for x ≤ 0 . Thus we consider x &gt; 0 in next two steps. Therefore we get σ ( wx ) = wx and b T a + x = ∑ i b i a + i x .

Step 2: Consider the random variables Z i = b i a + i . These are numbers of precision δ 2 , sampled from the set { ab | a, b ∈ S δ } . Now w , which is a number of precision δ t , also belongs to the set { ab | a, b ∈ S δ } because δ 2 ≤ δ t . The numbers Z i 's are not distributed uniformly, but by a standard rejection sampling argument (as in Lueker [1998]), there exists C such that more that 2 log 2 1 δ samples out of C log 2 1 δ are uniform distributed. We prune the other samples such that we are left with ¯ Z i , which are uniformly distributed. Now by Lemma 4, as long as cardinality of { ¯ Z i } is greater than 2 log 2 1 δ , the Random Subset Sum Problem with set { ¯ Z i } and target w can be solved with probability atleast

<!-- formula-not-decoded -->

Note that solving the Subset Sum Problem in an integer setting where numbers are sampled from 269 {-M,...,M } and solving it when numbers are sampled from {-1 , . . . δ, 2 δ, . . . , 1 } is equivalent 270 (only difference is a scaling factor). In Lemma 3 and 4, the sampling set is {-M,..., -1 , 1 , . . . , M } , 271 but 0 can be rejected during rejection sampling. Hence it follows that with probability p 272

<!-- formula-not-decoded -->

2 Without Loss of Generality

<!-- formula-not-decoded -->

Then the precision of elements of M i 's can be reduced to δ , such that for every f ∈ F ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where N t is the total number of parameters in f .

We prove the above theorem for a target network with a single weight (Lemma 5) using the results on RSSP in the previous section, and then we give the idea for proving it in general. The proof is an application of the strategy in Pensia et al. [2020] but with the use of Lemma 5. Details are given in Appendix B.

Lemma 5 (Representing a single weight) . Let g : R → R be a randomly initialized δ in quantized network of the form g ( x ) = [ v T σ ( u x )] γ where u , v , ∈ R 2 n . Assume δ 2 in ≤ δ t and δ 2 in ≤ δ 2 ≤ δ t . Also assume n &gt; C log 2 1 δ . Then the precision of weights of g can be reduced to δ , such that with probability atleast

<!-- formula-not-decoded -->

where S + δ denotes positive members of S δ . The part shown in green in Figure 1 in Appendix D hence 273 handles positive inputs. 274

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

Step 3: Similar to steps 1 and 2, we can prune negative weights from c and let the red part shown in Figure 1 in Appendix D handle negative inputs. It will follow that with probability p

<!-- formula-not-decoded -->

with probability p . Hence Lemma 4 follows.

Proof Sketch for Theorem 1. The idea is to use the strategy follow the strategy in Pensia et al. [2020]. We represented a single weight in Lemma 5. Similarly we can represent a neuron by representing each of its weights (shown explicitly in Lemma 6 and diagrammatically in Figure 2 in Appendix B). Using the representation of a single neuron, we represent a full layer (shown explicitly in Lemma 7 and diagrammatically in Figure 3 in Appendix B). Then we represent a full network by applying Lemma 7 layer by layer. See Appendix B for details.

Our next result employs construction from Burkholz [2022a].

Theorem 2. Let F be the class of δ t quantized γ -double mixed Precision neural networks of the form

<!-- formula-not-decoded -->

Consider an ℓ +1 layered randomly initialized γ -mixed precision resetting network which resets the precession to γ in all layers except the first one,

<!-- formula-not-decoded -->

whose weights are sampled from {-1 . . . , -δ, δ, . . . , 1 } with δ in ≤ δ t . Let δ in ≤ δ ≤ δ t . If M 1 and M 2 have dimensions

<!-- formula-not-decoded -->

respectively, M i +1 has dimension greater than

<!-- formula-not-decoded -->

∀ 2 &lt; i &lt; l -1 and M ℓ +1 has dimension greater than 291

<!-- formula-not-decoded -->

Then the precision elements of M i 's can be reduced to δ such that for every f ∈ F we have 292

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where N t is the total number of parameters in f .

293

294

Proof Sketch for Theorem 2. We follow the construction in Burkholz [2022a]. The idea is to use 295 the same trick as the previous result to represent a layer, but to copy it many times. Hence the 296 representation of a layer which was supposed to give output ( x 1 , x 2 , . . . , x N ) , will give output 297 ( x 1 , x 1 , . . . , x 2 , x 2 , . . . , x N , x N . . . , x N ) . These copies can now be used while representing the next 298 layer, without adding an intermediate layer in between (shown in Lemma 8 and diagrammatically in 299 Figure 4 in Appendix B). 300

with probability atleast

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

334

335

336

337

338

339

340

## 4.3 Lower Bound by Parameter Counting

In this section we show by a parameter counting argument, akin to that employed in Pensia et al. [2020], Natale et al. [2024], that there exists a two layered δ -quantized network with d 2 parameters that cannot be represented by a neural network with unless it has Ω ( d 2 log 2 ( 1 δ )) parameters. Note that any linear transformation Wx where W ∈ S d δ × S d δ and x ∈ S d δ can be expressed as a 2 layered neural network. Let F be the class of functions

<!-- formula-not-decoded -->

Theorem 3. Let g : R d → R d be a δ quantized neural network of the form

<!-- formula-not-decoded -->

where elements of M i 's are sampled from arbitrary distributions over S δ . Let G be the set of all matrices that can be formed by pruning g . Let F be defined as in Eq. 2. If

<!-- formula-not-decoded -->

then the total number of non zero parameters of g is at least

<!-- formula-not-decoded -->

Proof Sketch. Theorem 3 follows from a parameter counting argument. We simply count the number of different functions in F and demand that with probability p , any f ∈ F be represented by pruning g . See Appendix B for details.

The following immediate corollary of the previous theorem provide a matching lower bound to Theorem 1.

Corollary 1. If g is a two-layer network satisfying the hypothesis of Theorem 3, then its width is Ω( d log 1 δ ) .

## 5 Conclusion

We have proved optimal over-parameterization bounds for the Strong Lottery Ticket Hypothesis (SLTH) in the finite-precision setting. Specifically, we showed that any δ t -quantized target network N t can be recovered exactly by pruning a larger, randomly-initialized network N in with precision δ in . By reducing the pruning task to a quantized Random Subset Sum instance and importing the sharp phase-transition analysis for the Number Partitioning Problem, we derived width requirements that match the information-theoretic lower bound up to absolute constants. These results not only close the gap between upper and lower bounds for quantized SLTH, but also certify, for the first time, that pruning alone can yield exact finite-precision subnetworks rather than merely approximate ones. Beyond their theoretical interest, our findings pinpoint the precise interplay between quantization granularity and over-parameterization, and they suggest that mixed-precision strategies may enjoy similarly tight guarantees. An immediate open problem is to generalize our techniques to structured architectures-most notably convolutional, residual, and attention-based networks-where weight sharing and skip connections introduce additional combinatorial constraints. Another interesting direction is to incorporate layer-wise mixed precision and to analyze the robustness of lottery tickets under stochastic quantization noise, which is of interest for practical deployment on low-precision hardware accelerators. We believe that the combinatorial perspective adopted here will prove equally effective in these broader settings, ultimately advancing our theoretical understanding of extreme model compression.

## References

Christian Borgs, Jennifer Chayes, and Boris Pittel. Phase transition and finite-size scaling for the integer partitioning problem. Random Structures amp; Algorithms , 19(3-4):247-288, October 2001. ISSN 1098-2418. doi: 10.1002/rsa.10004. URL http://dx.doi.org/10.1002/rsa.10004 .

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

385

386

387

- Rebekka Burkholz. Most activation functions can win the lottery without excessive depth. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems , 2022a. URL https://openreview.net/forum?id= NySDKS9SxN .
- Rebekka Burkholz. Convolutional and Residual Networks Provably Contain Lottery Tickets. In Proceedings of the 39th International Conference on Machine Learning , pages 2414-2433, Baltimore, July 2022b. PMLR. URL https://proceedings.mlr.press/v162/burkholz22a.html .
- Michael Carilli. Automatic Mixed Precision - PyTorch Tutorials 2.7.0+cu126 documentation, 2020. URL https://docs.pytorch.org/tutorials/recipes/recipes/amp\_recipe.html .
- James Diffenderfer and Bhavya Kailkhura. Multi-prize lottery ticket hypothesis: Finding accurate binary neural networks by pruning a randomly weighted network. In International Conference on Learning Representations , 2021. URL https://openreview.net/forum?id=U\_mat0b9iv .
- Damien Ferbach, Christos Tsirigotis, Gauthier Gidel, and Joey Bose. A General Framework For Proving The Equivariant Strong Lottery Ticket Hypothesis. In The Eleventh International Conference on Learning Representations , September 2022. URL https://openreview.net/forum? id=vVJZtlZB9D .
- Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations , 2019. URL https: //openreview.net/forum?id=rJl-b3RcF7 .
- Michael R Garey and David S Johnson. Computers and intractability . W.H. Freeman, New York, NY, April 1979.
- Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding, 2015. URL https://arxiv.org/abs/ 1510.00149 .
- Brian Hayes. The easiest hard problem. American Scientist , 90(2):113, 2002. ISSN 1545-2786. doi: 10.1511/2002.2.113. URL http://dx.doi.org/10.1511/2002.2.113 .
- Geonho Hwang, Yeachan Park, and Sejun Park. On expressive power of quantized neural networks under fixed-point arithmetic. arXiv , 2024. URL https://arxiv.org/abs/2409.00297 .
- George S. Lueker. On the Average Difference between the Solutions to Linear and Integer Knapsack Problems. In Applied Probability-Computer Science: The Interface Volume 1 . Birkhäuser, 1982. ISBN 978-1-4612-5791-2. doi: 10.1007/978-1-4612-5791-2\_22. URL https://dl.acm.org/ doi/10.5555/313651.313692 .
- George S. Lueker. Exponentially small bounds on the expected optimum of the partition and subset sum problems. Random Structures and Algorithms , 12(1):51-62, January 1998. ISSN 1098-2418. URL http://dx.doi.org/10.1002/(SICI)1098-2418(199801)12:1&lt;51:: AID-RSA3&gt;3.0.CO;2-S .
- Eran Malach, Gilad Yehudai, Shai Shalev-Schwartz, and Ohad Shamir. Proving the lottery ticket hypothesis: Pruning is all you need. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 6682-6691. PMLR, 13-18 Jul 2020. URL https://proceedings. mlr.press/v119/malach20a.html .
- Stephan Mertens. Phase transition in the number partitioning problem. Physical Review Letters , 81(20):4281-4284, November 1998. ISSN 1079-7114. URL http://dx.doi.org/10.1103/ PhysRevLett.81.4281 .
- Marc Mézard and Andrea Montanari. Information, Physics, and Computation . Oxford University PressOxford, January 2009. ISBN 9780191718755. doi: 10.1093/acprof:oso/9780198570837.001. 0001. URL http://dx.doi.org/10.1093/acprof:oso/9780198570837.001.0001 .

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

- Emanuele Natale, Davide Ferre', Giordano Giambartolomei, Frédéric Giroire, and Frederik Mallmann-Trenn. On the Sparsity of the Strong Lottery Ticket Hypothesis. In The Thirtyeighth Annual Conference on Neural Information Processing Systems , November 2024. URL https://hal.science/hal-04741369v2 .
- Laurent Orseau, Marcus Hutter, and Omar Rivasplata. Logarithmic pruning is all you need. In Proceedings of the 34th International Conference on Neural Information Processing Systems , NIPS'20, pages 2925-2934, Red Hook, NY, USA, December 2020. Curran Associates Inc. ISBN 978-1-7138-2954-6. URL https://proceedings.neurips.cc/paper/2020/file/ 1e9491470749d5b0e361ce4f0b24d037-Paper.pdf .
- Ankit Pensia, Shashank Rajput, Alliot Nagle, Harit Vishwakarma, and Dimitris Papailiopoulos. Optimal lottery tickets via subset sum: Logarithmic over-parameterization is sufficient. In Advances in Neural Information Processing Systems , volume 33, pages 25992610, 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/file/ 1b742ae215adf18b75449c6e272fd92d-Paper.pdf .
- Vivek Ramanujan, Mitchell Wortsman, Aniruddha Kembhavi, Ali Farhadi, and Mohammad Rastegari. What's hidden in a randomly weighted neural network?, 2020. URL https://openaccess.thecvf.com/content\_CVPR\_2020/papers/Ramanujan\_Whats\_ Hidden\_in\_a\_Randomly\_Weighted\_Neural\_Network\_CVPR\_2020\_paper.pdf .
- Kartik Sreenivasan, Shashank Rajput, Jy-Yong Sohn, and Dimitris Papailiopoulos. Finding nearly everything within random binary networks. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera, editors, Proceedings of The 25th International Conference on Artificial Intelligence and Statistics , volume 151 of Proceedings of Machine Learning Research , pages 3531-3541. PMLR, 28-30 Mar 2022. URL https://proceedings.mlr.press/v151/sreenivasan22a.html .
- Yulong Wang, Xiaolu Zhang, Lingxi Xie, Jun Zhou, Hang Su, Bo Zhang, and Xiaolin Hu. Pruning from scratch, 2019. URL https://arxiv.org/abs/1909.12579 .
- Tim Dettmers Younes Belkada. A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using transformers, accelerate and bitsandbytes, 2022. URL https://huggingface.co/ blog/hf-bitsandbytes-integration .
- Hattie Zhou, Janice Lan, Rosanne Liu, and Jason Yosinski. Deconstructing lottery tickets: Zeros, signs, and the supermask. In Advances in Neural Information Processing Systems , pages 25992610, 2019. URL https://arxiv.org/abs/1905.01067 .

## A NPP and RSSP Results 419

We start by stating the result by Borgs et al. [2001] on NPP. Define I n,z as 420

<!-- formula-not-decoded -->

421

Theorem 4. Let C 0 &gt; 0 be a finite constant, let M = M ( n ) be an arbitrary function of n , let 422

<!-- formula-not-decoded -->

423

Furthermore 424

<!-- formula-not-decoded -->

if z and z ′ are of the same parity, i.e., both odd or both even, while E [ I n,z I n,z ′ ] = 0 if z and z ′ are of 425 different parity. 426

Now using 4, we prove 1 427

Proof of Lemma 1. Consider z = 0 . From Theorem 4 we have 428

̸

<!-- formula-not-decoded -->

If we multiply by 2 n +1 we get 429

<!-- formula-not-decoded -->

It also follows from the above equation that 430

<!-- formula-not-decoded -->

Furthermore, from Theorem 4 we have 431

<!-- formula-not-decoded -->

If z = z ′ we get 432

<!-- formula-not-decoded -->

Multiplying by (2 n +1 ) 2 we get 433

<!-- formula-not-decoded -->

and let z and z ′ be integers. Then,

<!-- formula-not-decoded -->

Now using Markov's inequality (Theorem 5, Appendix C) and Eq. 4 we get 434

<!-- formula-not-decoded -->

Using Cauchy-Schwartz inequality (Theorem 6, Appendix C) Eq. 5 and Eq. 6 we thus get 435

<!-- formula-not-decoded -->

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

From Lemma 1 it follows that

<!-- formula-not-decoded -->

Same can be done for z = 0 , only difference is a factor of 2. 461

The same calculation can be done for z = 0 , the only difference is that Z n,z = 2 n I n,z (Eq. 3).

Before moving ahead, lets establish the equivalence of NPP and SSP by proving Lemma 2.

Proof of Lemma 2. First of all notice that a NPP on the set X = ( X 1 , X 2 , . . . , X n ) where X i 's are sampled uniformly from {-M,..., -1 , 1 , . . . , M } can be solved iff the NPP on the set X = ( | X 1 | , | X 2 | , . . . , | X n | ) sampled uniformly from { 1 , 2 , . . . , M } can be solved. This is because, first, it is obvious that { X i } n i =1 is distributed uniformly over { 1 , 2 , . . . , M } , and secondly, the NPP does not care about the signs of the numbers, a sign can always be absorbed in the σ i while solving the NPP.

We have an SPP with set X , sampled uniformly from {-M,..., -1 , 1 , . . . , M } and target t . Assume number partitioning problem can be solved, given the set X and target Λ -2 t . Notice that NPP does not care about the sign of the target, as an NPP with target k can be solved iff that NPP with target -k can be solved. Assume there exists two partitions S 1 and S 2 of X , with S 1 summing to x and S 2 summing to Λ -x , such that ∑ i ∈ S 2 i -∑ j ∈ S 1 j = (Λ -x ) -x = Λ -2 , which is equivalent to ∑ j ∈ S 1 j = x = t . Hence, S 1 sums up to t , so the given SSP can be solved. The reverse direction also follows from the argument, proving the result.

̸

Proof of Lemma 3. Considers the number partitioning problem corresponding to the given random subset sum problem (Lemma 2). The target of this number partitioning problem is z = Λ -2 t . Consider z = 0 . A key observation here is if Λ is even (event denoted by E n ), then z is also even and if Λ is odd (event denoted by O n ), then z is also odd. The probability that the random subset sum problem can be solved can be written in terms of the probability that the number partitioning problem can be solved

<!-- formula-not-decoded -->

Since on E n , z is always even and on O n , z is always odd, we have two cases. If z is even, then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence P ( Y n,t &gt; 0) can be written as

<!-- formula-not-decoded -->

If z is odd, then

Proof of Lemma 4. We are given that lim n →∞ κ n exists and is less than 1. Consider a more sensitive 462 parametrization 463

<!-- formula-not-decoded -->

In this parametrization lim n →∞ κ n &lt; 1 means lim n →∞ λ n → -∞ . Note that in this regime 464 ρ n →∞ . Now we have 465

<!-- formula-not-decoded -->

Now t = O ( M ) and demand Λ as 466

472

473

474

475

476

477

478

479

480

we have 481

<!-- formula-not-decoded -->

According to Hoeffding's inequality (Theorem 7, Appendix C), that happens with probability 467

<!-- formula-not-decoded -->

Now as ρ n →∞ , we have 468

<!-- formula-not-decoded -->

Hence the probability (say P ( E ) ) of events P ( Y n,t &gt; 0) and Λ &lt; 1 √ 3+ β M √ n log n happening 469 together is given by 470

<!-- formula-not-decoded -->

Note that this probability will converge to 1 fastest if β = 1 2 . Hence we choose β = 1 2 and we get 471

<!-- formula-not-decoded -->

## B SLTH-Quantization Results

In this appendix, we prove the results related to SLTH and weight quantization. We start by proving Theorem 1. The idea is to follow the strategy of Pensia et al. [2020], but use Lemma 5.

Lemma 6 (Representing a single Neuron) . Consider a randomly initialized δ in quantized neural network of the form g ( x ) = [ v T σ ( M x )] γ with x ∈ R d . Assume δ 2 in ≤ δ t and δ 2 in ≤ δ 2 ≤ δ t . Let f w ( x ) = [ w T x ] γ be a single layered δ t quantized network. Let M ∈ R Cd log 2 1 δ × d and v ∈ R Cd log 2 1 δ . Then the precision of weights of g can be reduced to δ , such that with probability atleast

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where [ g { s , T } ] δ ( x ) is the pruned network for a choice of binary vector s and matrix T , 482

Proof. Assume weights of g are of precision δ . We prove the required results by representing each 483 weight of the neuron using Lemma 5 (See Figure 2, Appendix D). 484

Step 1: We first prune M to create a block-diagonal matrix M ′ . Specifically, we create M by only 485 keep the following non-zero entries: 486

<!-- formula-not-decoded -->

We choose the binary matrix T to be such that M ′ = T ⊙ M . We also decompose v and s as 487

<!-- formula-not-decoded -->

Step 2: Consider the event 488

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

<!-- formula-not-decoded -->

According to Lemma 5, this event happens with probability 489

<!-- formula-not-decoded -->

The event (say E ) in the assumption of Lemma 6 corresponds with the intersection of these events 490 E = ∩ d i =1 E i . By taking a union bound (Theorem 8, Appendix C), E happens with a probability 491 dp -( d -1) , which is equal to 492

<!-- formula-not-decoded -->

The process is illustrated in Figure 2. Note that we want &gt; log 2 ( 1 δ ) samples to be assured that a RSSP is solved with high probability, but we include that in the constant C . Any extra factors (a factor of 2 for example) is also absorbed in C throughout the proof.

Lemma 7 (Representing a single layer) . Consider a randomly initialized δ in quantized two layer neural network of the form g ( x ) = [ N σ ( Mx )] γ with x ∈ R d 1 . Assume δ 2 in ≤ δ t and δ 2 in ≤ δ 2 ≤ δ t . Let f W ( x ) = [ Wx ] γ be a single layered δ t quantized network. Assume N has dimension d 2 × Cd 1 log 2 1 δ and M has dimension Cd 1 log 2 1 δ × d 1 . Then the precision of weights of g can be reduced to to δ , such that with probability atleast

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where [ g { S , T } ] δ ( x ) is the pruned network for a choice of pruning matrices S and T .

Proof. Assume weights of g are of precision δ . We first prune M to get a block diagonal matrix M ′

<!-- formula-not-decoded -->

Thus, T is such that M ′ = T ⊙ M . We also decompose N and S as following 504

<!-- formula-not-decoded -->

Now note that pruning u i and v i,j (using s i,j ) is equivalent to Lemma 6. Hence it's simply an 505 application of Lemma 5 d 1 d 2 times. Hence the event in assumption of Lemma 7 occurs with a 506 probability d 1 d 2 p -( d 1 d 2 -1) , by a union bound (Theorem 8, Appendix C), which is equal to 507

<!-- formula-not-decoded -->

The process is illustrated in Figure 3, Appendix D. Note that we want &gt; log 2 ( 1 δ ) samples to be assured that a RSSP is solved with high probability, but we include that in the constant C . Constant Factors also absorbed in C .

Proof of Theorem 1. Now we can see that Theorem 1 can be proved by applying Lemma 7 layer wise, where two layers of the large network represent one layer of the target. Note that the precision is set of δ 1 after every layer (of the large network) and precision is set of δ 1 after every layer (of the target network). Let the total number of parameters in the target network be N t , i.e.,

<!-- formula-not-decoded -->

Then the event in assumption of Theorem 1, by union bound (Theorem 8, Appendix C), occurs with a 515 probability N t p -( N t -1) , where which is equal to 516

<!-- formula-not-decoded -->

## This construction improves the depth

In this subsection, we adapt construction by Burkholz [2022a] to prove Theorem 2. The process is illustrated in Figure 4, Appendix D.

Lemma 8. Consider a randomly initialized δ in quantized two layered neural network g ( x ) = [ N σ ( Mx )] γ with x ∈ R d 1 , whose weights are sampled uniformly from {-1 , . . . , -δ, δ, . . . , 1 } . Assume δ 2 in ≤ δ t and δ 2 in ≤ δ 2 ≤ δ t . Let

<!-- formula-not-decoded -->

be a single layered δ t quantized network where Wx is repeated log 2 ( 1 δ ) times and W has dimension 524 d 1 × d 2 . If N has dimension d 2 log 2 1 δ × Cd 1 log 2 1 δ and M has dimension Cd 1 log 2 1 δ × d 1 . Then 525 the precision of weights of g can be reduced to to δ , such that with probability 526

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have 527

where [ g { S , T } ] δ ( x ) is the pruned network for a choice of pruning matrices S and T . 528

Proof. Assume weights of g are of precision δ . We first prune M to get a block diagonal matrix M ′ 529

<!-- formula-not-decoded -->

508

509

510

511

512

513

514

517

518

519

520

521

522

523

Thus, T is such that M ′ = T ⊙ M . We also decompose N and S as following 530

<!-- formula-not-decoded -->

where 531

<!-- formula-not-decoded -->

Now note that pruning u i and ( v i,j ) k (using ( s i,j ) k ) is equivalent to Lemma 6. Hence it's simply 532 an application of Lemma 5 d 1 d 2 log 2 ( 1 δ ) times. Hence the event in assumption of Lemma 8 occurs 533 with a probability 534

<!-- formula-not-decoded -->

using the union bound (Theorem 8, Appendix C).

Proof of Theorem 2. In Lemma 8 we represented the first layer of the target network, with a difference that output contains many copies. The rest of the proof is same is Burkholz [2022a]. These copies can be used to represent weights in the next layer. The argument follows iteratively for all layers until we reach the last layer, where copying is not required. The only key difference is that rejection sampling is not required, giving the required size free of any undetermined constants. The process is illustrated in Figure 3. The event in the assumption of Theorem 2 happens with probability

<!-- formula-not-decoded -->

535

536

537

538

539

540

541

542

## Lower Bound by Parameter Counting 543

544

Here we prove Theorem 3 which follows by a parameter counting in the discrete setting.

Proof of Theorem 3. Two matrices represent the same function iff all their elements are the same. 545 Therefore, the number of functions in F is 546

<!-- formula-not-decoded -->

Let the number of non zero parameters in g be α , then the number of functions in G is 2 α . Now for 547 the assumption of Theorem 3 to hold, we must have 548

<!-- formula-not-decoded -->

549

Corollary 1 is an immediate consequence of Theorem 3. 550

## C Inequalities 551

Theorem 5. For a non-negative, integer-valued random variable X we have 552

<!-- formula-not-decoded -->

Theorem 6. If X &gt; 0 is a random variable with finite variance, then

<!-- formula-not-decoded -->

Theorem 7. Let X 1 , X 2 , . . . , X n be independent random variables such that a i ≤ X i ≤ b i almost surely. Consider the sum of these random variables,

<!-- formula-not-decoded -->

Then Hoeffding's theorem states that, for all t &gt; 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

553

554

555

556

557

558

559

Theorem 8. For any events A 1 , A 2 , . . . , A n we have 560

<!-- formula-not-decoded -->

561

## D Figures 562

Figure 1: Approximating a single weight with ReLU activation (Pensia et al. [2020]): The network shown in the figure represents a single weight after pruning.

<!-- image -->

Figure 2: Representing a single neuron (Pensia et al. [2020]): The figure on the left shows the target network, where as Figure on the right shows the large network. The colors indicate which part in the target is represented by which part of the source. For example, the red weight on the left is represented by the red subnetwork on the right.

<!-- image -->

Figure 3: Representing a layer (Pensia et al. [2020]): Figure on the left shows the target network, where as Figure on the right shows the large network. The colors indicate which part in the target is represented by which part of the source. For example, the red weight on the left is represented by the red weights on the right.

<!-- image -->

Figure 4: The Figure shows representation of first two layers of a network in Theorem 2: (Burkholz [2022a]). The figure on the left shows the target network, where as Figure on the right shows the large network. The colors indicate which part in the target is represented by which part of the source. For example, the red weight on the left is represented by the red weights on the right.

<!-- image -->

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

575

576

577

578

579

580

581

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

598

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

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We provide complete proofs to the results stated in the abstract.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We sate our assumptions clearly, and we discuss where the assumptions might be unnatural, but were made for theoretical simplicity.

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

Justification: We provide complete proofs to all the results in supplemental material. All Theorems are well numbered and cross referenced.

## Guidelines:

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

720

721

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

Justification: The authors have read the NeurIPS Code of Ethics and conform, in every respect with it.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

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

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA] .

Justification: LLMs were used only for grammatical corrections in this work.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.
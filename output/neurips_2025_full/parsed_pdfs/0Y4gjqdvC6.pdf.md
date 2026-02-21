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

31

32

## Fundamental Limits of Game-Theoretic LLM Alignment: Smith Consistency and Preference Matching

## Anonymous Author(s) Affiliation Address email

## Abstract

Nash Learning from Human Feedback ( NLHF ) is a game-theoretic framework for aligning large language models (LLMs) with human preferences by modeling learning as a two-player zero-sum game. However, using raw preference as the payoff in the game highly limits the potential of the game-theoretic LLM alignment framework. In this paper, we systematically study using what choices of payoff based on the pairwise human preferences can yield desirable alignment properties. We establish necessary and sufficient conditions for Condorcet consistency, diversity through mixed strategies, and Smith consistency. These results provide a theoretical foundation for the robustness of game-theoretic LLM alignment. Further, we show the impossibility of preference matching-i.e., no smooth and learnable mappings of pairwise preferences can guarantee a unique Nash equilibrium that matches a target policy, even under standard assumptions like the Bradley-Terry-Luce (BTL) model. This result highlight the fundamental limitation of game-theoretic LLM alignment.

## 1 Introduction

Large language models (LLMs), such as OpenAI-o3 (OpenAI, 2025) and DeepSeek-R1 (DeepSeekAI et al., 2025), have demonstrated impressive capabilities across a wide range of domains, including code generation, data analysis, elementary mathematics, and reasoning (Hurst et al., 2024; Anthropic, 2024; Chowdhery et al., 2023; Touvron et al., 2023; Ji et al., 2025). These models are increasingly being used to tackle previously unsolved mathematical problems, drive scientific and algorithmic discoveries, optimize complex code-bases, and support decision-making processes that were once considered unlikely to be automated in the near future (Bubeck et al., 2023; Eloundou et al., 2024; Novikov et al., 2025).

A key factor behind the popularity and effectiveness of LLMs is alignment: the process by which models learn to interact with human users and accommodate diverse human opinions and values by aligning their outputs with human preferences (Christiano et al., 2017). The traditional method for alignment, reinforcement learning from human feedback ( RLHF ) (Ouyang et al., 2022; Casper et al., 2023; Dong et al., 2024), typically begins by training a reward model on preference data collected from human labelers, often using the Bradley-Terry-Luce (BTL) model (Bradley and Terry, 1952; Luce, 2012),

<!-- formula-not-decoded -->

where r ( x, y ) is the reward function and P ( y ≻ y ′ | x ) is pairwise human preference, i.e., the fraction of individuals who prefer y over y ′ under prompt x . In this framework, a higher scalar score assigned

by the reward model to an LLM-generated response indicates a stronger preference by human 33 labelers. The LLM is then fine-tuned through maximizing the reward to produce responses that are 34 more likely to align with these preferences. However, Munos et al. (2024) points out that reward 35 model can not deal with preference with cycles, and proposes an alternative alignment approach 36 called Nash learning from human feedback ( NLHF ). Unlike the reward-based methods, NLHF directly 37 uses preference data to train a preference model and formulates LLM finetuning as finding Nash 38 equilibrium in a two-player zero-sum game, also known as a von Neumann game (Myerson, 2013). 39 Specifically, for a given prompt x , the LLM's policy π competes against an opposing policy π ′ in 40 a pairwise preference contest, where the objective is to find a policy that maximizes its worst-case 41 preference score. Formally, NLHF solves the following min-max optimization problem: 42

<!-- formula-not-decoded -->

where ρ is a given distribution over prompts. However, Munos et al. (2024) does not demonstrate the advantages of using the preference as the payoff in the game.

Recently, criteria from both social choice theory (Conitzer et al., 2024; Dai and Fleisig, 2024; Mishra, 2023) and principles related to diversity (Xiao et al., 2024; Chakraborty et al., 2024) has been increasingly employed to scrutinize the alignment of LLM with human preference. Notably, RLHF has been shown to fail both social choice theory considerations (Noothigattu et al., 2020; Siththaranjan et al., 2024; Ge et al., 2024; Liu et al., 2025) and diversity considerations (Xiao et al., 2024; Chakraborty et al., 2024). In contrast, NLHF has been proved to enjoy these desirable properties. It is shown in Maura-Rivero et al. (2025); Liu et al. (2025) that NLHF is Condorcet consistent (see Definition 3.2), meaning that the method always outputs the Condorcet winning response, a response that beats every other alternative response in pairwise majority comparisons, whenever one exists. Further, under a no-tie assumption (see Assumption 2.1), Liu et al. (2025) shows that NLHF is Smith consistent (see Definition 4.1), meaning that the method always output responses from the Smith set, the smallest nonempty set of responses that pairwise dominate all alternatives outside the set. Moreover, Liu et al. (2025) shows that when human preference is diverse, i.e., there does not exist a single response that beat every other alternative, NLHF avoid collapsing to a single response by adopting a mixed strategy .

Despite these advantages, there is no reason why we must use raw preference to design the payoff in the game-theoretic alignment approach. Using alternative payoffs in the game-theoretic LLM alignment framework might also lead to desirable alignment. In this work, we systematically investigate the fundamental limits of the game-theoretic LLM alignment framework by analyzing how various choices of payoff influence its ability to satisfy key alignment criteria. We consider the following general game-theoretic alignment problem, involving a mapping of the preference denoted by Ψ :

<!-- formula-not-decoded -->

The general problem (1.2) encompasses a range of games. When Ψ( t ) = t is the identity mapping, the objective in Equation (1.2) is equivalent to the standard NLHF objective. When Ψ( t ) = log( t/ (1 -t )) and the preference is generated by a BTL model, Equation (1.2) recovers the standard RLHF objective. More importantly, the preference model P θ used in practice is an estimation of true human preference, which can be regarded as a noisy mapping of the ground-truth preference P . Allowing Ψ to be stochastic provides a way to account for the uncertainty and noise inherent in estimating human preferences. It is worthy to note that similar formalism has been proposed for non game-theoretic approach in Azar et al. (2024), which uses an non-decreasing mapping to process the preference.

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

In this paper, we first discuss when the solution to problem (1.2) is Condorcet consistent and Smith consistent in Section 3 and 4 respectively. Our results show that these desirable properties is insensitive to the exact value of the payoff, revealing the robustness of game-theoretic alignment approaches. As a special case, we discover a natural generalization of RLHF objective that satisfy all these desirable properties. Technically, we develop novel proof techniques that can tackle a general non-symmetric game directly, instead of relying crucially on the symmetric nature of NLHF as in Liu et al. (2025).

In addition, we examine the diversity of the solution by investigating whether the model produces 82 a mixed strategy (Liu et al., 2025), and whether its output can satisfy the criterion of preference 83 matching (Xiao et al., 2024), meaning that the model output exactly matches a target policy which 84

- fully accounts for the diversity of human preference. Our findings suggest diversity can be ensured by 85

mixed strategies, but exactly matching a target is difficult for any game-theoretic alignment approach. 86

This reveal a fundamental limitation of game-theoretic alignment approaches. 87

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

## 1.1 Summary of Contributions

We summarize our contributions as follows:

- We show that Condorcet consistency is insensitive to the exact value of the payoff (Theorem 3.1), revealing the robustness of game-theoretic alignment approaches.
- We show that Smith consistency can be ensured by further maintaining the symmetry of the game (Theorem 4.2). Moreover, Smith consistent methods automatically preserve the diversity in human preferences by adopting mixed strategies (Corollary 4.2).
- We show that preserving the diversity in human preference strictly, in the sense of preference matching, is impossible in general (Theorem 5.1). This reveals a fundamental limitation of game-theoretic alignment approaches.

Assuming Ψ is continuous, Table 1 provides a concise summary of our mathematical results. 98

Table 1: Summary of our mathematical results: the necessary and sufficient conditions on Ψ to guarantee certain desirable alignment properties.

| Condorcet consistency        | Ψ( t ) ⩾ Ψ(1 / 2) , ∀ 1 / 2 ⩽ t ⩽ 1 and Ψ( t ) < Ψ(1 / 2) , ∀ 0 ⩽ t < 1 / 2             |
|------------------------------|-----------------------------------------------------------------------------------------|
| Mixed &Condorcet consistency | Ψ( t ) +Ψ(1 - t ) ⩾ 2Ψ(1 / 2) , ∀ 1 / 2 ⩽ t ⩽ 1 and Ψ( t ) < Ψ(1 / 2) , ∀ 0 ⩽ t < 1 / 2 |
| Smith consistency            | Ψ( t ) +Ψ(1 - t ) = 2Ψ(1 / 2) , ∀ 1 / 2 ⩽ t ⩽ 1 and Ψ( t ) < Ψ(1 / 2) , ∀ 0 ⩽ t < 1 / 2 |

## 1.2 Related Works 99

100

101

102

103

104

105

106

107

108

A general mapping Ψ is first introduced in Azar et al. (2024) to facilitate the analysis of traditional non game-theoretic LLM alignment methodologies. Their objective function, called Ψ PO , applies a general mapping Ψ to the original human preference. In this way, they treat RLHF and DPO as special cases of Ψ PO under BTL model and argue that they are prone to overfitting. To avoid overfitting, they take Ψ to be identity and arrive at a new efficient algorithm called IPO . Our problem (1.2) can be regarded as the analogy of Ψ PO in the context of game-theoretic LLM alignment. Another difference is that rather than focusing on statistical properties like overfitting, our focus is on the alignment properties such as Smith consistency and preference matching. Moreover, they restrict Ψ to be an non-decreasing map, while we allow Ψ to be arbitrary, even stochastic.

109

110

111

112

113

114

115

116

117

Condorcet consistency is one of the dominant concept in the theory of voting (Gehrlein, 2006; Balinski and Laraki, 2010), and Smith consistency is its natural generalization (Shoham and Leyton-Brown, 2008; Börgers, 2010). They are not studied in the context of LLM alignment until recently (MauraRivero et al., 2025; Liu et al., 2025). In Maura-Rivero et al. (2025), the authors show that NLHF with a selection probability that deals with ties is Condorcet consistent. Under a no-tie assumption, Liu et al. (2025) show that NLHF is Condorcet consistent and Smith consistent, whereas RLHF is not unless the preference satisfies a BTL model. Further, they show that the probability that the preference satisfies a BTL model is vanishing under an impartial culture assumption, highlighting a key advantage of the NLHF framework.

Several recent works also focus on aligning LLMs with the diverse human preference (Chakraborty 118 et al., 2024; Xiao et al., 2024; Liu et al., 2025). In Chakraborty et al. (2024), the authors introduce a 119 mixture model to account for the opinion of minority group and arrive at the MaxMin-RLHF method. 120 In Xiao et al. (2024), the authors introduce the concept of preference matching and develop the 121 PM-RLHF objective to pursue this goal. Liu et al. (2025) demonstrates that the original NLHF yields 122 a mixed strategy when no Condorcet winning response exists, whereas standard RLHF produces a 123 deterministic strategy, highlighting a potential advantage of NLHF in preserving the diversity of human 124

preferences. 125

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

167

168

169

170

171

172

## 2 Preliminaries

Consider a general mapping Ψ : [0 , 1] → R . We apply Ψ to the preference and study the max-min problem (1.2) with this generalized payoff. Any solution π employed by the first player at the Nash equilibrium,

<!-- formula-not-decoded -->

is called a Nash solution to the problem (1.2). The Nash solution is the policy which fully aligned LLMs will perform. Note that the set of Nash solutions remain the same after an overall shift of payoff, that is, changing Ψ to Ψ+ C for any constant C will not affect the problem. The original NLHF objective (Munos et al., 2024) corresponds to the special case where Ψ( t ) = t , equivalent to Ψ( t ) = t -1 / 2 , and the resulting game is symmetric (Duersch et al., 2012), meaning that the two players are the same. However, for an arbitrary mapping Ψ , the game is usually not symmetric, and we only focus on the Nash solution employed by the first player.

Given a prompt x , we consider the set of all possible responses generated by the LLM: { y 1 , . . . , y n } , where n is the total number of possible responses. Without any loss of generality, we drop the dependence on the prompt x from now on. For any two distinct response y and y ′ , recall that P ( y ≻ y ′ ) denote the preference of y over y ′ , defined as the expected proportion of individuals who prefer y over y ′ . By definition, human preference satisfies the condition P ( y ≻ y ′ ) + P ( y ′ ≻ y ) = 1 and naturally we let P ( y ≻ y ) = 1 / 2 (Munos et al., 2024). For any distinct pair of responses y and y ′ , we say that y beats y ′ if P ( y ≻ y ′ ) &gt; 1 / 2 . Additionally, following Liu et al. (2025), we adopt the No-Tie assumption throughout this paper.

̸

Assumption 2.1 (No-Tie) . For any distinct responses y and y ′ , we assume that P ( y ≻ y ′ ) = 1 / 2 .

This assumption is both minimal and practically reasonable. First, if the number of labelers is odd, it automatically holds. Even in cases where a tie occurs, it can always be resolved through a more precise comparison.

Notation. For any set A , we denote its cardinality by | A | . For any n ∈ N + , we define [ n ] := { 1 , . . . , n } . We use δ ij := 1 { i = j } for 1 ⩽ i, j ⩽ n . We represent high-dimensional vectors using bold symbols. Any policy π over the set of possible responses { y 1 , . . . , y n } can be identified with a vector in R n , where each entry π i corresponds to the probability assigned to y i for i ∈ [ n ] . We then define the support of a policy π as supp( π ) := { y i | π i &gt; 0 , i ∈ [ n ] } . We write π &gt; 0 if π i &gt; 0 for all i ∈ [ n ] , and similarly, π ⩾ 0 if π i ⩾ 0 for all i ∈ [ n ] .

## 3 Condorcet Consistency

In this section, we examine Condorcet consistency-a desirable property for LLM alignment inspired by social choice theory-within the generalized game-theoretic LLM fine-tuning framework (1.2). We begin by defining the Condorcet winning response and Condorcet consistency. We then present Theorem 3.1, which characterizes the necessary and sufficient conditions on the mapping Ψ to guarantee Condorcet consistency. Next, we examine the conditions under which Ψ preserves human preference diversity when no Condorcet winner exists and introduce Theorem 3.2. Finally, we discuss the continuity assumption underlying Theorem 3.2.

Following Liu et al. (2025), a response that is preferred over all others in pairwise comparisons by the preference model is referred to as the Condorcet winning response.

̸

Definition 3.1 (Condorcet Winning Response) . Aresponse y ⋆ is called a Condorcet winning response if P ( y ⋆ ≻ y ) &gt; 1 / 2 for all y = y ⋆ .

It is clear that there can be at most one Condorcet winning response. When such a response exists, a natural requirement for LLM alignment is that this response should be the output. This property is known as Condorcet consistency.

Definition 3.2 (Condorcet Consistency) . Problem (1.2) is Condorcet consistent if when there exists a Condorcet winning response, the Nash solution to (1.2) is unique and corresponds to this Condorcet winning response.

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

212

213

214

215

216

Liu et al. (2025); Maura-Rivero et al. (2025) show that the original NLHF objective, which corresponds to the case where Ψ( · ) is identity, is Condorcet consistent. In this paper, we proceed further and investigate the following question:

Which choices of Ψ ensure Condorcet consistency?

We answer this question in Theorem 3.1. The proof is provided in Appendix A.

Theorem 3.1. Problem (1.2) is Condorcet consistent if and only if Ψ( · ) satisfies

<!-- formula-not-decoded -->

Note that this condition is much weaker than requiring Ψ to be increasing. It only demands that Ψ maps any value greater than 1/2 to some value larger than Ψ(1 / 2) , and any value less than 1 / 2 to some value smaller than Ψ(1 / 2) . This implies that a wide range of mapping functions can be used within the game-theoretic LLM alignment framework (1.2) to ensure Condorcet consistency. Furthermore, in practice, we do not have access to the ground-truth preference model. Instead, we parameterize the preference model using a deep neural network, P θ ( y ≻ y ′ ) , trained on large-scale preference datasets (Munos et al., 2024). Due to the limitations of the datasets and the optimization process, the learned model only approximates the true human preferences. We can view this approximation as Ψ( P ( y ≻ y ′ )) in our framework. In practice, we can enforce the parameterized preference model to satisfy P θ ( y ≻ y ) = 1 / 2 , then our results show that as long as this approximation yields the correct pairwise majority comparisons-specifically, that P θ ( y ≻ y ′ ) ⩾ 1 / 2 &gt; P θ ( y ′ ≻ y ) whenever y beats y ′ -then the LLM alignment remains Condorcet consistent. This strongly highlights the robustness of the game-theoretic LLM alignment approach in achieving Condorcet consistency.

When a Condorcet winning response does not exist, human preferences are diverse and there is no single response that is better than others. Therefore, in order to preserve the diversity inherent in human preferences, it is natural to require the Nash solution not to collapse to a single response. This motivation leads to the following characterization of diversity through mixed strategies.

Definition 3.3 (Mixed Strategies) . A Nash solution π is called a mixed strategy if | supp( π ) | &gt; 1 .

Liu et al. (2025) demonstrates that the original NLHF , which corresponds to the case where Ψ( · ) is identity, yields a mixed strategy when no Condorcet winning response exists. Assuming that problem (1.2) is Condorcet consistent, we proceed further and investigate:

Which choices of Ψ lead to a mixed strategy in the absence of a Condorcet winning response?

We now focus on mappings Ψ that are continuous at 1/2, a condition commonly encountered in practical learning setups. Under this mild assumption, we answer this question in Theorem 3.2 and the proof is provided in Appendix C.

Theorem 3.2. Assume that the mapping Ψ( · ) is continuous at 1 / 2 . Assuming the Condorcet consistency of problem (1.2) , then any Nash solution is mixed when there is no Condorcet winning response if and only if Ψ( · ) satisfies

<!-- formula-not-decoded -->

The first condition arises from the requirement of mixed strategies, while the second condition is a reduction of the condition inherited from Theorem 3.1 under the assumption of Condorcet consistency and the first condition.

Choices of payoff functions are harder to characterize when we relax the continuity assumption. The following example investigate a special piece-wise constant mapping, which does not satisfy the first condition in Theorem 3.2.

Example 3.4. Let M -&lt; Ψ(1 / 2) ⩽ M + and take

<!-- formula-not-decoded -->

Then, any Nash solution is mixed when there is no Condorcet winning response.

The proof of Example 3.4 is deferred to Appendix B. This example implies that choices of payoff functions are considerably richer when we relax the continuity assumption.

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

255

256

257

258

259

260

## 4 Smith Consistency

In this section, we extend the discussion of Condorcet consistency to Smith consistency. First, we define the Smith set and Smith consistency. Next, we present Theorem 4.2, which provides the necessary and sufficient condition for the mapping Ψ to ensure Smith consistency. Finally, we highlight that Smith-consistent methods inherently preserve the diversity present in human preferences and discuss the continuity assumption in Theorem 4.2.

Condorcet consistency only ensures the method capture the right response when there exists a Condorcet winning response. In general, when there is no Condorcet winning response, we can expect that there might be a set of responses satisfying similar property, generalizing Definition 3.1. Under Assumption 2.1, Liu et al. (2025) revealed a more detailed decomposition of the preference structure. Specifically, the set of responses can be partitioned into distinct groups S 1 , . . . , S k , where every response in S i is preferred over all responses in S j for i &lt; j , summarized in the following theorem.

Theorem 4.1 (Liu et al. (2025)) . Under Assumption 2.1, the set of responses can be partitioned into disjoint subsets S 1 , . . . , S k such that:

1. Each S i either forms a Condorcet cycle or is a single response.
2. For any j &gt; i , any response y ∈ S i and y ′ ∈ S j , P ( y ≻ y ′ ) &gt; 1 2 .

Moreover, this decomposition is unique.

When | S 1 | = 1 , the response in S 1 is exactly the Condorcet winning response. Thus, S 1 is the generalization of Condorcet winning response, and is referred as the Condorcet winning set in Liu et al. (2025). Traditionally, a subset with such property is also known as the Smith set in the literature of social choice theory (Shoham and Leyton-Brown, 2008). Here we choose to adopt the name Smith set to distinguish with the concept of Condorcet winning response. Given this decomposition, it is natural to desire that an aligned LLM adopts a strategy supported exclusively on the top group S 1 , as any response outside S 1 is strictly less preferred than any response inside S 1 . This desirable property is referred to as Smith consistency:

Definition 4.1 (Smith Consistency) . Problem (1.2) is Smith consistent if the support of any Nash solution is contained in the Smith set S 1 .

Liu et al. (2025) showed that the original NLHF payoff, which corresponds to the case where Ψ( t ) = t , is Smith consistent. Here, we investigate this question for a general mapping Ψ :

Which choices of Ψ ensure Smith consistency?

Here, similar to Theorem 3.2, we answer this question in Theorem 4.2 for mappings that is continuous at 1 / 2 . The proof is provided in Appendix E.

Theorem 4.2. Suppose that the mapping Ψ( · ) is continuous at 1 / 2 , problem (1.2) is Smith consistent if and only if Ψ( · ) satisfies

<!-- formula-not-decoded -->

The first condition Ψ( t ) + Ψ(1 -t ) = 2Ψ(1 / 2) says nothing but the zero-sum game formed by problem (1.2) is equivalent to a symmetric two-player zero-sum game 1 (Duersch et al., 2012). By definition, Smith consistency implies Condorcet consistency because when there is a Condorcet winning response, S 1 is exactly the set whose only element is the Condorcet winning response. Thus, the second condition is just a reduction of the condition in Theorem 3.1 under the first condition. It is easy to see Ψ( t ) = t satisfies these conditions, and thus our result generalize Theorem 3.6 in Liu et al. (2025). More interestingly, Ψ( t ) = log( t/ (1 -t )) also satisfies these conditions. This implies that

<!-- formula-not-decoded -->

which is a natural generalization of standard RLHF when human preferences does not satisfy BTL model, is also Smith consistent.

1 This can be seen by shifting the payoff by Ψ(1 / 2) , which leaves the Nash solution unchanged.

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

The set of choices for Ψ that ensure Smith consistency is quite broad. We can easily construct such a Ψ by first defining Ψ( t ) on [0 , 1 / 2] to satisfy Ψ( t ) &lt; Ψ(1 / 2) for all t ∈ [0 , 1 / 2) , and then extending it to [0 , 1] by setting Ψ( t ) = 2Ψ(1 / 2) -Ψ(1 -t ) for all t ∈ (1 / 2 , 1] . Moreover, as discussed in Section 3, a practical preference model P θ ( y ≻ y ′ ) can be seen as a mapping of the ground truth preference via Ψ , i.e., Ψ( P ( y ≻ y ′ )) . Thus, the first condition in Theorem 4.2 requires the preference model to satisfy P θ ( y ≻ y ′ ) + P θ ( y ′ ≻ y ) = 1 , with P θ ( y ≻ y ) = 1 / 2 enforced. However, several practically used preference models (Munos et al., 2024; Jiang et al., 2023; Wu et al., 2024) do not guarantee this condition, which may cause the aligned LLM strategy to fail to satisfy Smith consistency.

As any mapping satisfying the condition in Theorem 4.2 also satisfies the condition in Theorem 3.2, we obtain the following corollary:

Corollary 4.2. Suppose that the mapping Ψ( · ) is continuous at 1 / 2 . Then if problem (1.2) is Smith consistent, any Nash solution is also mixed.

This shows that when | S 1 | &gt; 1 , the Nash solution to problem (1.2) with any Ψ such that Smith consistency holds will not only support on S 1 but also be a mixed strategy on S 1 without collapsing to a single response. As a conclusion, a Smith consistent method can preserve the diversity inherent in human preferences, at least partially.

Lastly, we discuss what happens if Ψ is not continuous at 1 / 2 . Choices of mappings Ψ are considerably richer and consequently harder to characterize when we relax the continuity assumption. The following example shows that the piece-wise constant mapping in Example 3.4 also ensures Smith consistency.

Example 4.3. Let M -&lt; Ψ( 1 2 ) &lt; M + , and we take

<!-- formula-not-decoded -->

Then problem (1.2) is Smith consistent. The proof is provided in Appendix D.

## 5 Impossibility of Preference Matching

In this section, we first revisit the definition of preference matching in the BTL model (Xiao et al., 2024). Then we introduce a general theoretical framework of preference matching within the context of game-theoretic LLM alignment, and establish a general impossibility result, as stated in Theorem 5.1. Finally we apply this general result to problem (1.2), concluding that preference matching is impossible.

In previous sections, we have characterized the diversity of alignment result via mixed strategies. In Xiao et al. (2024), the authors propose a more refined criterion for diversity when the preference P ( y ≻ y ′ | x ) follows a BTL model,

<!-- formula-not-decoded -->

as given by (1.1). They point out that it is unwise to completely disregard any minority opinions in the case that 51% of human labelers prefer y 1 over y 2 for a binary comparison. They suggest that the policy (5.1),

<!-- formula-not-decoded -->

referred to as the preference-matching policy, fully accounts for the diversity in human preferences.

It is easy to see that there exists a Condorcet winning response under BTL model. According to Theorem 3.1, using preference Ψ( P ( y ≻ y ′ | x )) as payoff with Ψ( t ) = t or Ψ( t ) = log( t/ (1 -t )) will lead the Nash solution to collapse to a single response instead of matching with π ∗ . This shows that both RLHF and NLHF do not accounts for the diversity inherent in human preferences from the perspective of preference matching (Xiao et al., 2024; Liu et al., 2025), even under BTL model.

To achieve alignment fully accounting for diversity, we would like to match the Nash solution with the desired policy π ∗ . In Xiao et al. (2024), the authors answered this question for RLHF . They proposed

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

341

the preference matching RLHF ( PM-RLHF ) method which successfully achieves preference matching, by slightly modifying the RLHF objective. Here, we aim to explore the possibility of designing a new learnable payoff matrix that aligns with the desired strategy in a game-theoretic framework for LLM alignment:

## Which choices of Ψ ensure preference matching?

Although it is currently unknown how to generalize the notion of preference matching policy to a general non-BTL preference, to maintain the generality of the discussion and drop the BTL model assumption, we suppose there exists an ideal policy, denoted by π ∗ , which captures the diversity of human preferences perfectly.

Given a prompt x , we consider the set of all possible responses generated by the LLM: { y 1 , . . . , y n } . We further suppose that the policy π ∗ has full support over these n responses, meaning π ∗ &gt; 0 , as we exclude responses not supported by π ∗ from consideration. Then our goal is to construct a game, represented by a payoff matrix { α ij } n i,j =1 , with its Nash solution the given policy π ∗ , i.e.,

<!-- formula-not-decoded -->

To answer this question, we characterize the Nash solution under the given payoff matrix { α ij } n i,j =1 , which is summarized by the following KKT condition. The proof is deferred to Appendix F.1.

Lemma 5.1 (KKT Condition) . Consider a game with payoff matrix { α ij } n i,j =1 . Then π ∗ &gt; 0 is a Nash solution to the game if and only if there exists u ∗ ∈ R n with u ∗ ⩾ 0 and ∑ n i =1 u ∗ i = 1 , and t ∗ ∈ R such that the following KKT conditions hold:

<!-- formula-not-decoded -->

According to Lemma 5.1, it is easy to verify that the payoff matrix

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

both guarantee that π ∗ is a Nash solution (the details are provided in Appendix F.2). However, these payoff matrices do not depend on the given policy π ∗ in a reasonable way. The payoff matrix in Equation (5.2) is symmetric, making it difficult to interpret. Even worse, it depends on the raw value of π ∗ . In practice, π ∗ is often only known up to a normalizing constant. For instance, the preference matching policy (5.1) includes a normalizing constant in the denominator that involves summing over n terms. This constant is hard to determine when n is large and unknown, as is often the case in LLMs. The payoff matrix in Equation (5.3) faces a similar issue as it explicitly depends on n , which is an extremely large and unknown value in practice.

In summary, the above two payoff matrices rely on information that is often unavailable in practice, such as n and the raw value of π ∗ . What we can obtain in practice for the design of α ij is the preference information between two responses y i and y j , which we assume depends solely on the ratio between π ∗ i and π ∗ j . When the preference satisfies the BTL model (1.1), this assumption is justified by the fact that the preference between any two responses depends solely on the ratio of the values assigned by their corresponding preference matching policies (5.1). From this practical consideration, we assume that the payoff matrix satisfies the following assumptions:

Assumption 5.2. Given any π ∗ &gt; 0 , the payoff matrix { α ij } n i,j =1 satisfies the following conditions:

1. For all i ∈ [ n ] , α ii = C where C is a constant independent of π ∗ and n . In other words, the diagonal elements are the same constant.

and the payoff matrix

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

388

̸

̸

2. For all i, j ∈ [ n ] with i = j , α ij = f ( π ∗ i π ∗ j ) for some smooth function f that is independent of π ∗ and n . In other words, the off-diagonal elements depend on the ratio π ∗ i π ∗ j in the same way for all pairs ( i, j ) with i = j .

Weemphasize that the above two assumptions are crucial for constructing a meaningful and practically learnable payoff matrix. Furthermore, for effective alignment, the payoff matrix should not only ensure π ∗ to be a Nash solution, but π ∗ must be the only Nash solution. The uniqueness requirement excludes trivial payoff matrices such as α ij = C , where every π ∗ &gt; 0 is a Nash solution.

Unfortunately, in Theorem 5.1, we prove that such a payoff matrix { α ij } n i,j =1 does not exist generally. The proof can be found in Appendix F.3.

Theorem 5.1 (Impossibility of Preference Matching for General Payoffs) . There does not exist a payoff matrix { α ij } n i,j =1 satisfying Assumption 5.2 such that for any given π ∗ &gt; 0 , the Nash solution to the game is unique and equals to π ∗ .

Remark 5.3. If we relax Assumption 5.2 and allow the entries of the payoff matrix to depend on n , then the design (5.3) is actually eligible for preference matching.

Theorem 5.1 implies that no simple mapping of the preference can yield a payoff that leads to preference matching. As a special case of Theorem 5.1, under BTL model, the generalized game in Equation (1.2) with a smooth mapping Ψ ,

<!-- formula-not-decoded -->

cannot achieve preference matching. Therefore, we obtain the following corollary.

Corollary 5.4. Problem (1.2) with smooth mapping Ψ cannot achieve preference matching.

## 6 Conclusion

We investigate several properties motivated by social choice theory and diversity considerations within the general game-theoretic LLM alignment framework (1.2), where the payoff is designed as a mapping Ψ of the original preference. We identify the necessary and sufficient conditions on Ψ to guarantee Condorcet consistency and Smith consistency. These conditions allow for a considerably broad class of choices for Ψ , demonstrating that these desirable alignment properties are not sensitive to the exact values of the payoff, thereby providing a theoretical foundation for the robustness of the game-theoretic LLM alignment approach. Additionally, we examine conditions on Ψ that ensure the resulting policy is a mixed strategy, preserving diversity in human preferences. Finally, we prove that achieving exact preference matching is impossible under the general game-theoretic alignment framework with a smooth mapping, revealing fundamental limitations of this approach.

Limitations and Discussion. Our findings suggest several promising directions for future research on LLM alignment. First, while we establish an impossibility result for preference matching under the assumption that Ψ is smooth, it remains an open question whether preference matching can be achieved when Ψ is merely continuous. Second, in practical settings, regularization terms based on the reference model are often added to problem (1.2). Regularization may be crucial for preference matching, for example, Xiao et al. (2024) modify the regularization term in RLHF to achieve preference matching. Analyzing the alignment properties of game-theoretic methods with such regularization is another interesting avenue for future work. Furthermore, how to explicitly define a preferencematching policy for general preferences that do not satisfy the BTL model, and how to develop alignment approaches capable of learning such a policy, remain open problems. Finally, our results highlight that practical preference models must satisfy certain anti-symmetry conditions to ensure Smith consistency - conditions that are not guaranteed by several currently used models. Thus, designing preference model architectures that enforce anti-symmetry is an important and interesting future direction.

Broader Impacts. The goal of this paper is to investigate several theoretical properties of the general game-theoretic LLM alignment approach. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

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

434

435

436

437

438

## References

- Anthropic, A. (2024). The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card .
- Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M., and Calandriello, D. (2024). A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics , pages 4447-4455. PMLR.
- Balinski, M. L. and Laraki, R. (2010). Majority judgment : measuring, ranking, and electing . MIT Press.
- Börgers, C. (2010). Mathematics of social choice: voting, compensation, and division . Society for Industrial and Applied Mathematics.
- Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika , 39(3/4):324-345.
- Bubeck, S., Chandrasekaran, V., Eldan, R., Gehrke, J., Horvitz, E., Kamar, E., Lee, P., Lee, Y. T., Li, Y., Lundberg, S., Nori, H., Palangi, H., Ribeiro, M. T., and Zhang, Y. (2023). Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712 .
- Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., Freedman, R., Korbak, T., Lindner, D., Freire, P., Wang, T. T., Marks, S., Segerie, C.-R., Carroll, M., Peng, A., Christoffersen, P. J., Damani, M., Slocum, S., Anwar, U., Siththaranjan, A., Nadeau, M., Michaud, E. J., Pfau, J., Krasheninnikov, D., Chen, X., Langosco, L., Hase, P., Biyik, E., Dragan, A., Krueger, D., Sadigh, D., and Hadfield-Menell, D. (2023). Open problems and fundamental limitations of reinforcement learning from human feedback. Transactions on Machine Learning Research .
- Chakraborty, S., Qiu, J., Yuan, H., Koppel, A., Huang, F., Manocha, D., Bedi, A. S., and Wang, M. (2024). Maxmin-rlhf: Towards equitable alignment of large language models with diverse human preferences. arXiv preprint arXiv:2402.08925 .
- Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., Schuh, P., Shi, K., Tsvyashchenko, S., Maynez, J., Rao, A., Barnes, P., Tay, Y., Shazeer, N., Prabhakaran, V., Reif, E., Du, N., Hutchinson, B., Pope, R., Bradbury, J., Austin, J., Isard, M., Gur-Ari, G., Yin, P., Duke, T., Levskaya, A., Ghemawat, S., Dev, S., Michalewski, H., Garcia, X., Misra, V., Robinson, K., Fedus, L., Zhou, D., Ippolito, D., Luan, D., Lim, H., Zoph, B., Spiridonov, A., Sepassi, R., Dohan, D., Agrawal, S., Omernick, M., Dai, A. M., Pillai, T. S., Pellat, M., Lewkowycz, A., Moreira, E., Child, R., Polozov, O., Lee, K., Zhou, Z., Wang, X., Saeta, B., Diaz, M., Firat, O., Catasta, M., Wei, J., Meier-Hellstern, K., Eck, D., Dean, J., Petrov, S., and Fiedel, N. (2023). Palm: Scaling language modeling with pathways. Journal of Machine Learning Research , 24(240):1-113.
- Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. (2017). Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems , volume 30.
- Conitzer, V., Freedman, R., Heitzig, J., Holliday, W. H., Jacobs, B. M., Lambert, N., Mossé, M., Pacuit, E., Russell, S., Schoelkopf, H., Tewolde, E., and Zwicker, W. S. (2024). Position: social choice should guide ai alignment in dealing with diverse human feedback. In Forty-first International Conference on Machine Learning .
- Dai, J. and Fleisig, E. (2024). Mapping social choice theory to RLHF. In ICLR 2024 Workshop on Reliable and Responsible Foundation Models .
- DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J.-M., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y ., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B.-L., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D.-L., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q.,

Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R. J., Jin, R., Chen, R., Lu, S., Zhou, S., Chen, S., 439 Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S.-K., Yun, T., Pei, T., Sun, T., 440 Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W.-X., Zhang, W., Xiao, W. L., 441 An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, 442 X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X.-C., Chen, X., Sun, X., Wang, X., Song, X., Zhou, 443 X., Wang, X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang, Y., Xu, Y., Li, Y ., Zhao, Y ., Sun, 444 Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y ., Xiong, Y ., He, Y ., Piao, Y ., Wang, Y ., Tan, Y ., Ma, Y ., Liu, 445 Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y.-J., He, Y ., Xiong, Y ., Luo, Y .-W., mei You, Y ., Liu, 446 Y., Zhou, Y., Zhu, Y. X., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., 447 Ren, Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., guo Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, 448 Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z.-A., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., 449 and Zhang, Z. (2025). Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement 450 learning. arXiv preprint arXiv:2501.12948 . 451

452

453

454

- Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou, Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. (2024). RLHF workflow: From reward modeling to online RLHF. Transactions on Machine Learning Research .

455

456

457

458

459

460

461

462

- Duersch, P., Oechssler, J., and Schipper, B. C. (2012). Pure strategy equilibria in symmetric two-player zero-sum games. International Journal of Game Theory , 41:553-564.
- Eloundou, T., Manning, S., Mishkin, P., and Rock, D. (2024). GPTs are GPTs: Labor market impact potential of LLMs. Science , 384(6702):1306-1308.
- Ge, L., Halpern, D., Micha, E., Procaccia, A. D., Shapira, I., Vorobeychik, Y., and Wu, J. (2024).
- Axioms for ai alignment from human feedback. In Advances in Neural Information Processing Systems , volume 37, pages 80439-80465.

Gehrlein, W. V. (2006). Condorcet's paradox . Springer.

Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., 463 Hayes, A., Radford, A., M ˛ adry, A., Baker-Whitcomb, A., Beutel, A., Borzunov, A., Carney, A., 464 Chow, A., Kirillov, A., Nichol, A., Paino, A., Renzin, A., Passos, A. T., Kirillov, A., Christakis, 465 A., Conneau, A., Kamali, A., Jabri, A., Moyer, A., Tam, A., Crookes, A., Tootoochian, A., 466 Tootoonchian, A., Kumar, A., Vallone, A., Karpathy, A., Braunstein, A., Cann, A., Codispoti, A., 467 Galu, A., Kondrich, A., Tulloch, A., Mishchenko, A., Baek, A., Jiang, A., Pelisse, A., Woodford, A., 468 Gosalia, A., Dhar, A., Pantuliano, A., Nayak, A., Oliver, A., Zoph, B., Ghorbani, B., Leimberger, 469 B., Rossen, B., Sokolowsky, B., Wang, B., Zweig, B., Hoover, B., Samic, B., McGrew, B., Spero, 470 B., Giertler, B., Cheng, B., Lightcap, B., Walkin, B., Quinn, B., Guarraci, B., Hsu, B., Kellogg, B., 471 Eastman, B., Lugaresi, C., Wainwright, C., Bassin, C., Hudson, C., Chu, C., Nelson, C., Li, C., 472 Shern, C. J., Conger, C., Barette, C., V oss, C., Ding, C., Lu, C., Zhang, C., Beaumont, C., Hallacy, 473 C., Koch, C., Gibson, C., Kim, C., Choi, C., McLeavey, C., Hesse, C., Fischer, C., Winter, C., 474 Czarnecki, C., Jarvis, C., Wei, C., Koumouzelis, C., Sherburn, D., Kappler, D., Levin, D., Levy, 475 D., Carr, D., Farhi, D., Mely, D., Robinson, D., Sasaki, D., Jin, D., Valladares, D., Tsipras, D., Li, 476 D., Nguyen, D. P., Findlay, D., Oiwoh, E., Wong, E., Asdar, E., Proehl, E., Yang, E., Antonow, 477 E., Kramer, E., Peterson, E., Sigler, E., Wallace, E., Brevdo, E., Mays, E., Khorasani, F., Such, 478 F. P., Raso, F., Zhang, F., von Lohmann, F., Sulit, F., Goh, G., Oden, G., Salmon, G., Starace, 479 G., Brockman, G., Salman, H., Bao, H., Hu, H., Wong, H., Wang, H., Schmidt, H., Whitney, H., 480 Jun, H., Kirchner, H., de Oliveira Pinto, H. P., Ren, H., Chang, H., Chung, H. W., Kivlichan, I., 481 O'Connell, I., O'Connell, I., Osband, I., Silber, I., Sohl, I., Okuyucu, I., Lan, I., Kostrikov, I., 482 Sutskever, I., Kanitscheider, I., Gulrajani, I., Coxon, J., Menick, J., Pachocki, J., Aung, J., Betker, 483 J., Crooks, J., Lennon, J., Kiros, J., Leike, J., Park, J., Kwon, J., Phang, J., Teplitz, J., Wei, J., 484 Wolfe, J., Chen, J., Harris, J., Varavva, J., Lee, J. G., Shieh, J., Lin, J., Yu, J., Weng, J., Tang, J., Yu, 485 J., Jang, J., Candela, J. Q., Beutler, J., Landers, J., Parish, J., Heidecke, J., Schulman, J., Lachman, 486 J., McKay, J., Uesato, J., Ward, J., Kim, J. W., Huizinga, J., Sitkin, J., Kraaijeveld, J., Gross, J., 487 Kaplan, J., Snyder, J., Achiam, J., Jiao, J., Lee, J., Zhuang, J., Harriman, J., Fricke, K., Hayashi, 488 K., Singhal, K., Shi, K., Karthik, K., Wood, K., Rimbach, K., Hsu, K., Nguyen, K., Gu-Lemberg, 489 K., Button, K., Liu, K., Howe, K., Muthukumar, K., Luther, K., Ahmad, L., Kai, L., Itow, L., 490 Workman, L., Pathak, L., Chen, L., Jing, L., Guy, L., Fedus, L., Zhou, L., Mamitsuka, L., Weng, L., 491 McCallum, L., Held, L., Ouyang, L., Feuvrier, L., Zhang, L., Kondraciuk, L., Kaiser, L., Hewitt, 492 L., Metz, L., Doshi, L., Aflak, M., Simens, M., Boyd, M., Thompson, M., Dukhan, M., Chen, 493

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

M., Gray, M., Hudnall, M., Zhang, M., Aljubeh, M., Litwin, M., Zeng, M., Johnson, M., Shetty, M., Gupta, M., Shah, M., Yatbaz, M., Yang, M. J., Zhong, M., Glaese, M., Chen, M., Janner, M., Lampe, M., Petrov, M., Wu, M., Wang, M., Fradin, M., Pokrass, M., Castro, M., de Castro, M. O. T., Pavlov, M., Brundage, M., Wang, M., Khan, M., Murati, M., Bavarian, M., Lin, M., Yesildal, M., Soto, N., Gimelshein, N., Cone, N., Staudacher, N., Summers, N., LaFontaine, N., Chowdhury, N., Ryder, N., Stathas, N., Turley, N., Tezak, N., Felix, N., Kudige, N., Keskar, N., Deutsch, N., Bundick, N., Puckett, N., Nachum, O., Okelola, O., Boiko, O., Murk, O., Jaffe, O., Watkins, O., Godement, O., Campbell-Moore, O., Chao, P., McMillan, P., Belov, P., Su, P., Bak, P., Bakkum, P., Deng, P., Dolan, P., Hoeschele, P., Welinder, P., Tillet, P., Pronin, P., Tillet, P., Dhariwal, P., Yuan, Q., Dias, R., Lim, R., Arora, R., Troll, R., Lin, R., Lopes, R. G., Puri, R., Miyara, R., Leike, R., Gaubert, R., Zamani, R., Wang, R., Donnelly, R., Honsby, R., Smith, R., Sahai, R., Ramchandani, R., Huet, R., Carmichael, R., Zellers, R., Chen, R., Chen, R., Nigmatullin, R., Cheu, R., Jain, S., Altman, S., Schoenholz, S., Toizer, S., Miserendino, S., Agarwal, S., Culver, S., Ethersmith, S., Gray, S., Grove, S., Metzger, S., Hermani, S., Jain, S., Zhao, S., Wu, S., Jomoto, S., Wu, S., Shuaiqi, Xia, Phene, S., Papay, S., Narayanan, S., Coffey, S., Lee, S., Hall, S., Balaji, S., Broda, T., Stramer, T., Xu, T., Gogineni, T., Christianson, T., Sanders, T., Patwardhan, T., Cunninghman, T., Degry, T., Dimson, T., Raoux, T., Shadwell, T., Zheng, T., Underwood, T., Markov, T., Sherbakov, T., Rubin, T., Stasi, T., Kaftan, T., Heywood, T., Peterson, T., Walters, T., Eloundou, T., Qi, V., Moeller, V., Monaco, V., Kuo, V., Fomenko, V., Chang, W., Zheng, W., Zhou, W., Manassra, W., Sheu, W., Zaremba, W., Patil, Y., Qian, Y., Kim, Y., Cheng, Y., Zhang, Y., He, Y., Zhang, Y., Jin, Y., Dai, Y., and Malkov, Y. (2024). Gpt-4o system card. arXiv preprint arXiv:2410.21276 .

- Ji, W., Yuan, W., Getzen, E., Cho, K., Jordan, M. I., Mei, S., Weston, J. E., Su, W. J., Xu, J., and Zhang, L. (2025). An overview of large language models for statisticians. arXiv preprint arXiv:2502.17814 .
- Jiang, D., Ren, X., and Lin, B. Y. (2023). LLM-blender: Ensembling large language models with pairwise ranking and generative fusion. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 14165-14178. Association for Computational Linguistics.
- Liu, K., Long, Q., Shi, Z., Su, W. J., and Xiao, J. (2025). Statistical impossibility and possibility of aligning llms with human preferences: From condorcet paradox to nash equilibrium. arXiv preprint arXiv:2503.10990 .
- Luce, R. D. (2012). Individual choice behavior: A theoretical analysis . Courier Corporation.
- Maura-Rivero, R.-R., Lanctot, M., Visin, F., and Larson, K. (2025). Jackpot! alignment as a maximal lottery. arXiv preprint arXiv:2501.19266 .
- Mishra, A. (2023). Ai alignment and social choice: Fundamental limitations and policy implications. arXiv preprint arXiv:2310.16048 .
- Munos, R., Valko, M., Calandriello, D., Gheshlaghi Azar, M., Rowland, M., Guo, Z. D., Tang, Y., Geist, M., Mesnard, T., Fiegel, C., Michi, A., Selvi, M., Girgin, S., Momchev, N., Bachem, O., Mankowitz, D. J., Precup, D., and Piot, B. (2024). Nash learning from human feedback. In Forty-first International Conference on Machine Learning .
- Myerson, R. B. (2013). Game theory . Harvard university press.
- Noothigattu, R., Peters, D., and Procaccia, A. D. (2020). Axioms for learning from pairwise comparisons. In Advances in Neural Information Processing Systems , volume 33, pages 1774517754.
- Novikov, A., V˜ u, N., Eisenberger, M., Dupont, E., Huang, P.-S., Wagner, A. Z., Shirobokov, S., Kozlovskii, B., Ruiz, F. J. R., Mehrabian, A., Kumar, M. P., See, A., Chaudhuri, S., Holland, G., Davies, A., Nowozin, S., Kohli, P., and Balog, M. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery. Technical report, Google DeepMind.
- OpenAI (2025). Openai o3 and o4-mini system card. Technical report, OpenAI.

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

- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., and Lowe, R. (2022). Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems , volume 35, pages 27730-27744.
- Shoham, Y. and Leyton-Brown, K. (2008). Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations . Cambridge University Press.
- Siththaranjan, A., Laidlaw, C., and Hadfield-Menell, D. (2024). Distributional preference learning: Understanding and accounting for hidden context in RLHF. In The Twelfth International Conference on Learning Representations .
- Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. (2023). Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 .
- Wu, Y., Sun, Z., Yuan, H., Ji, K., Yang, Y., and Gu, Q. (2024). Self-play preference optimization for language model alignment. In Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning .
- Xiao, J., Li, Z., Xie, X., Getzen, E., Fang, C., Long, Q., and Su, W. J. (2024). On the algorithmic bias of aligning large language models with rlhf: Preference collapse and matching regularization. arXiv preprint arXiv:2405.16455 .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims in the abstract and introduction accurately reflect the contributions and scope of this paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses the limitations of the work in Section 6 (see the "Limitations and Discussion" paragraph).

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

## Answer: [Yes]

Justification: The full set of assumptions is introduced before theoretical result and the complete proof of theoretical result is provided in the appendix.

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

Justification: This paper focuses on theoretical analysis and does not include experiments.

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

Justification: This paper does not include experiments requiring code.

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

Justification: This paper does not include experiments.

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

765

766

767

768

769

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

Justification: Our research conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper discusses broader impacts in Section 6 (see the "Broader Impacts" paragraph).

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

Justification: This paper poses no risk for misuse.

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

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Proof of Theorem 3.1 885

Notation. For simplicity, we denote Ψ( P ( y i ≻ y j )) as Ψ ij for any 1 ⩽ i, j ⩽ n , and define the 886 payoff matrix as Ψ := Ψ ij 1 ⩽ i,j ⩽ n . We then define the total payoff by: 887

<!-- formula-not-decoded -->

We denote by δ i the policy supported solely on y i , i.e., supp( δ i ) = { y i } . The mixed policy 888 ( δ i 1 + . . . + δ i k ) /k is then defined as the policy π such that 889

<!-- formula-not-decoded -->

for any subset { i 1 , . . . , i k } ⊆ [ n ] .

Proof of Theorem 3.1. Without any loss of generality, we assume that y 1 is the Condorcet winning response. First, we show that a necessary condition that ensures the Condorcet consistency of problem (1.2) is:

<!-- formula-not-decoded -->

To show this, we examine the case where n = 2 . For any 1 ⩾ t &gt; 1 / 2 , we consider the game with 894 the payoff in Table 2. By the definition of Condorcet consistency, all Nash equilibrium of this game 895 is of the form ( δ 1 , π ⋆ ) for some π ⋆ . 896

Table 2: Payoff matrix with two responses { y 1 , y 2 } .

| Ψ( P ( y ≻ y ′ )) y ′   | = y 1     | y ′ = y 2       |
|-------------------------|-----------|-----------------|
| y = y 1                 | Ψ(1 / 2)  | Ψ( t ) Ψ(1 / 2) |
| y = y 2                 | Ψ(1 - t ) |                 |

Case 1. If Ψ( t ) &gt; Ψ(1 / 2) , we have 897

<!-- formula-not-decoded -->

898

Therefore, we have

<!-- formula-not-decoded -->

Hence, we have Ψ(1 -t ) ⩽ Ψ(1 / 2) &lt; Ψ( t ) . If Ψ(1 / 2) = Ψ(1 -t ) , notice that 899

<!-- formula-not-decoded -->

Therefore, ( δ 2 , δ 1 ) is also a Nash equilibrium, which causes a contradiction to the fact that problem 900 (1.2) is Condorcet consistent. Therefore, we have Ψ( t ) &gt; Ψ(1 / 2) &gt; Ψ(1 -t ) for any 1 ⩾ t &gt; 1 / 2 . 901

902

903

Case 2. If Ψ( t ) &lt; Ψ(1 / 2) , we have

<!-- formula-not-decoded -->

However, notice that

<!-- formula-not-decoded -->

which causes a contradiction. 904

890

891

892

893

Case 3. If Ψ( t ) = Ψ(1 / 2) . When Ψ(1 -t ) = Ψ(1 / 2) , any ( π 1 , π 2 ) is a Nash equilibrium, which 905 causes a contradiction to the fact that problem (1.2) is Condorcet consistent. When Ψ(1 -t ) &gt; 906 Ψ(1 / 2) , note that 907

<!-- formula-not-decoded -->

Therefore, ( δ 2 , δ 2 ) is a Nash equilibrium, which also causes a contradiction to the fact problem (1.2) 908 is Condorcet consistent. Hence, we have Ψ(1 -t ) &lt; Ψ(1 / 2) . 909

910

911

912

913

In summary, for any 1 ⩾ t &gt; 1 / 2 , we have Ψ(1 -t ) &lt; Ψ(1 / 2) ⩽ Ψ( t ) . Hence, (A.1) holds if problem (1.2) is Condorcet consistent. Next, we prove that (A.1) is also sufficient for the Condorcet consistency of problem (1.2). Recall that Ψ i 1 = Ψ( P ( y i ≻ y 1 )) &lt; Ψ(1 / 2) , and Ψ 1 i = Ψ( P ( y 1 ≻ y i )) ⩾ Ψ(1 / 2) for any i = 1 . If ( π ⋆ 1 , π ⋆ 2 ) is a Nash equilibrium. Notice that

̸

<!-- formula-not-decoded -->

̸

Therefore, if π ⋆ 1 = δ 1 , we have 914

<!-- formula-not-decoded -->

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

which causes a contradiction. Therefore, π ⋆ 1 = δ 1 , i.e., problem (1.2) is Condorcet consistent. Hence, we conclude our proof.

## B Proof of Example 3.4

Proof of Example 3.4. We prove this conclusion by contradiction. Suppose that the Nash solution is δ i ⋆ for some i ⋆ ∈ [ n ] , and the Nash equilibrium is ( δ i ⋆ , π ⋆ ) . As there is no Condorcet winning response, by definition, there exists j ′ such that P ( y i ⋆ ≻ y j ′ ) &lt; 1 / 2 . Then we have

<!-- formula-not-decoded -->

However, choosing i ′ such that π ⋆ i ′ &gt; 0 , we have

<!-- formula-not-decoded -->

which causes a contradiction to (B.1). Hence, we conclude our proof.

## C Proof of Theorem 3.2

Proof of Theorem 3.2. First, according to Theorem 3.1, when the Nash solution is Condorcet consistent, we have

<!-- formula-not-decoded -->

In addition, we show that Ψ( · ) must satisfy Ψ( t ) + Ψ(1 -t ) ⩾ 2Ψ(1 / 2) , ∀ t ∈ [0 , 1] for ensuring 926 that the Nash solution is mixed when there is no Condorcet winning response. We consider the 927 case where n = 4 and the game with the payoff in Table 3 for any t 1 , t 2 &gt; 1 / 2 . Notice that if 928 Ψ( t 1 ) + Ψ(1 -t 1 ) + Ψ(1 / 2) ⩽ 3Ψ(1 -t 2 ) , we have 929

<!-- formula-not-decoded -->

and 930

<!-- formula-not-decoded -->

Therefore, ( δ 4 , ( δ 1 + δ 2 + δ 3 ) / 3) is a Nash equilibrium, which causes a contradiction to the fact 931 that the Nash solution is mixed. Hence, we have Ψ( t 1 ) + Ψ(1 -t 1 ) + Ψ(1 / 2) &gt; 3Ψ(1 -t 2 ) for 932 any t 1 , t 2 &gt; 1 / 2 . Let t 2 → 1 / 2 , we have Ψ( t ) + Ψ(1 -t ) ⩾ 2Ψ(1 / 2) for any t ∈ [0 , 1] . Hence, 933 combining (C.1), we have shown that the necessary condition for ensuring that the Nash solution is 934 mixed is: 935

<!-- formula-not-decoded -->

Next, we prove that the condition (C.2) is also sufficient. Suppose that ( δ i ⋆ , π ⋆ ) is a Nash equilibrium, 936 then we have 937

<!-- formula-not-decoded -->

However, notice that for any j , we have 938

<!-- formula-not-decoded -->

As there is no Condorcet winning response, there must exist j ⋆ such that P ( y i ⋆ ≻ y j ⋆ ) &lt; 1 / 2 , thus Ψ i ⋆ j ⋆ &lt; Ψ(1 / 2) . Hence, Ψ(1 / 2) ⩽ P Ψ ( δ i ⋆ , π ⋆ ) ⩽ Ψ i ⋆ j ⋆ &lt; Ψ(1 / 2) , which causes a contradiction. Therefore, the Nash solution must be mixed.

## D Proof of Example 4.3

Proof of Example 4.3. We prove this conclusion by contradiction. Suppose that the Nash solution is π ⋆ 1 that satisfies supp( π ⋆ 1 ) ∩ S c 1 = ∅ , and the Nash equilibrium is ( π ⋆ 1 , π ⋆ 2 ) .

̸

Case 1. If supp( π ⋆ 1 ) ∩ S 1 = ∅ , taking j ′ ∈ S 1 , we have

939

940

941

942

943

944

945

<!-- formula-not-decoded -->

However, we have 946

<!-- formula-not-decoded -->

947

948

which causes a contradiction.

Case 2. If supp( π ⋆ 2 ) ∩ S 1 = ∅ and supp( π ⋆ 1 ) ∩ S 1 = ∅ , taking i ′ ∈ supp( π ⋆ 1 ) ∩ S 1 , we have

̸

<!-- formula-not-decoded -->

However, we have 949

<!-- formula-not-decoded -->

which cause a contradiction. 950

Then we have 953

<!-- formula-not-decoded -->

where the last inequality follows from the following two facts: for any i ∈ S c 1 , 954

<!-- formula-not-decoded -->

955

and when j = i ⋆ 2 ,

<!-- formula-not-decoded -->

However, (D.1) causes a contradiction to the fact that P Ψ ( π ′ 1 , π ⋆ 2 ) ⩽ max π P Ψ ( π , π ⋆ 2 ) = 956 P Ψ ( π ⋆ 1 , π ⋆ 2 ) . 957

958

Hence, in summary, it must hold that supp( π ⋆ 1 ) ∩ S c 1 = ∅ , i.e., supp( π ⋆ 1 ) ⊆ S 1 .

959

960

961

## E Proof of Theorem 4.2

Proof of Theorem 4.2. First, we show that the necessary condition for ensuring that problem (1.2) is Smith consistent is:

<!-- formula-not-decoded -->

First, Condorcet consistency must hold when Smith consistency holds. According to Theorem 3.1, 962 we have 963

<!-- formula-not-decoded -->

Next, we show that when Ψ( · ) is continuous at 1 / 2 , Ψ( · ) must satisfy Ψ( t ) + Ψ(1 -t ) ⩾ 2Ψ(1 / 2) 964 (Lemma E.1) and Ψ( t ) + Ψ(1 -t ) ⩽ 2Ψ(1 / 2) (Lemma E.2) for any t ∈ [0 , 1] . Therefore, combining 965 the two results together, we obtain the condition (E.1). 966

Lemma E.1. When Ψ( · ) is continuous at 1 / 2 . Achieving Smith consistency only if 967

<!-- formula-not-decoded -->

Proof of Lemma E.1. We consider the case where n = 4 and the game with the payoff in Table 3 for 968 any t 1 , t 2 &gt; 1 / 2 . Notice that if Ψ( t 1 ) + Ψ(1 -t 1 ) + Ψ(1 / 2) ⩽ 3Ψ(1 -t 2 ) , we have 969

<!-- formula-not-decoded -->

̸

Case 3. If If supp( π ⋆ 2 ) ∩ S 1 = ∅ and supp( π ⋆ 1 ) ∩ S 1 = ∅ , taking i ⋆ 2 ∈ supp( π ⋆ 2 ) ∩ S 1 , we consider 951 the following strategy π ′ 1 : 952

̸

<!-- formula-not-decoded -->

and 970

<!-- formula-not-decoded -->

Therefore, ( δ 4 , ( δ 1 + δ 2 + δ 3 ) / 3) is a Nash equilibrium, which causes a contradiction to the fact that 971 the Nash solution supports on S 1 := { y 1 , y 2 , y 3 } . Hence, we have Ψ( t 1 ) + Ψ(1 -t 1 ) + Ψ(1 / 2) &gt; 972 3Ψ(1 -t 2 ) for any t 1 , t 2 &gt; 1 / 2 . Let t 2 → 1 / 2 , we have Ψ( t ) + Ψ(1 -t ) ⩾ 2Ψ(1 / 2) for any 973 t ∈ [0 , 1] .

Table 3: Payoff matrix with four responses { y 1 , y 2 , y 3 , y 4 } .

| Ψ( P ( y ≻ y ′ ))   | y ′ = y 1   | y ′ = y 2   | y ′ = y 3   | y ′ = y 4   |
|---------------------|-------------|-------------|-------------|-------------|
| y = y 1             | Ψ(1 / 2)    | Ψ( t 1 )    | Ψ(1 - t 1 ) | Ψ( t 2 )    |
| y = y 2             | Ψ(1 - t 1 ) | Ψ(1 / 2)    | Ψ( t 1 )    | Ψ( t 2 )    |
| y = y 3             | Ψ( t 1 )    | Ψ(1 - t 1 ) | Ψ(1 / 2)    | Ψ( t 2 )    |
| y = y 4             | Ψ(1 - t 2 ) | Ψ(1 - t 2 ) | Ψ(1 - t 2 ) | Ψ(1 / 2)    |

974

Lemma E.2. When Ψ( · ) is continuous at 1 / 2 . Achieving Smith consistency only if 975

<!-- formula-not-decoded -->

Proof of Lemma E.2. We consider the case where n = 6 and the game with the payoff in 976 Table 4 for any t 1 , t 2 &gt; 1 / 2 . Notice that if Ψ( t 1 ) + Ψ(1 / 2) + Ψ(1 -t 1 ) &gt; 3Ψ( t 2 )( ⩾ 977 3Ψ(1 / 2) &gt; 3Ψ(1 -t 2 )) , there exists positive µ = ( µ 1 / 3 , µ 1 / 3 , µ 1 / 3 , µ 2 / 3 , µ 2 / 3 , µ 2 / 3) and 978 µ ′ = ( µ ′ 1 / 3 , µ ′ 1 / 3 , µ ′ 1 / 3 µ ′ 2 / 3 , µ ′ 2 / 3 , µ ′ 2 / 3) such that µ 1 + µ 2 = µ ′ 1 + µ ′ 2 = 1 , and 979

<!-- formula-not-decoded -->

Hence, we have 980

Thus, we have 981

<!-- formula-not-decoded -->

and 982

<!-- formula-not-decoded -->

Therefore, µ ∈ arg max π P Ψ ( π , µ ′ ) , µ ′ ∈ arg min π P Ψ ( µ , π ) , which provides that ( µ , µ ′ ) is a 983 Nash equilibrium. However, this causes a contradiction to the fact that the Nash solution supports 984 on S 1 := { y 1 , y 2 , y 3 } . Thus, it must hold that Ψ( t 1 ) + Ψ(1 / 2) + Ψ(1 -t 1 ) ⩽ 3Ψ( t 2 ) for any 985 t 1 , t 2 &gt; 1 / 2 . Let t 2 → 1 / 2 , we obtain Ψ( t ) + Ψ(1 -t ) ⩽ 2Ψ(1 / 2) for any t ∈ [0 , 1] . 986

<!-- formula-not-decoded -->

Then we have 995

<!-- formula-not-decoded -->

which follows from the following fact: 996

<!-- formula-not-decoded -->

However, (E.3) also causes a contradiction to the fact that P Ψ ( π ⋆ 1 , π ⋆ 2 ) = Ψ(1 / 2) . 997

Therefore, it must hold that supp( π ⋆ 1 ) ⋂ S c 1 = ∅ , i.e., supp( π ⋆ 1 ) ⊆ S 1 . We conclude our proof. 998

Table 4: Payoff matrix with six responses { y 1 , y 2 , y 3 , y 4 , y 5 , y 6 } .

| Ψ( P ( y ≻ y ′ ))   | y ′ = y 1   | y ′ = y 2   | y ′ = y 3   | y ′ = y 4   | y ′ = y 5   | y ′ = y 6   |
|---------------------|-------------|-------------|-------------|-------------|-------------|-------------|
| y = y 1             | Ψ(1 / 2)    | Ψ( t 1 )    | Ψ(1 - t 1 ) | Ψ( t 2 )    | Ψ( t 2 )    | Ψ( t 2 )    |
| y = y 2             | Ψ(1 - t 1 ) | Ψ(1 / 2)    | Ψ( t 1 )    | Ψ( t 2 )    | Ψ( t 2 )    | Ψ( t 2 )    |
| y = y 3             | Ψ( t 1 )    | Ψ(1 - t 1 ) | Ψ(1 / 2)    | Ψ( t 2 )    | Ψ( t 2 )    | Ψ( t 2 )    |
| y = y 4             | Ψ(1 - t 2 ) | Ψ(1 - t 2 ) | Ψ(1 - t 2 ) | Ψ(1 / 2)    | Ψ( t 1 )    | Ψ(1 - t 1 ) |
| y = y 5             | Ψ(1 - t 2 ) | Ψ(1 - t 2 ) | Ψ(1 - t 2 ) | Ψ(1 - t 1 ) | Ψ(1 / 2)    | Ψ( t 1 )    |
| y = y 6             | Ψ(1 - t 2 ) | Ψ(1 - t 2 ) | Ψ(1 - t 2 ) | Ψ( t 1 )    | Ψ(1 - t 1 ) | Ψ(1 / 2)    |

Finally, we prove that the condition (E.1) is also sufficient for Smith consistency. Suppose that 987 ( π ⋆ 1 , π ⋆ 2 ) is a Nash equilibrium, notice that 988

<!-- formula-not-decoded -->

which follows from the following fact: for any π , 989

<!-- formula-not-decoded -->

Thus, from (E.2), we have P Ψ ( π ⋆ 1 , π ⋆ 2 ) = Ψ(1 / 2) . Then we prove supp( π ⋆ 1 ) ⊆ S 1 . Hence, the Nash 990 solution is Smith consistent, i.e., only supports on S 1 . 991

Case 1. If supp( π ⋆ 1 ) ⋂ S 1 = ∅ , taking any j ∈ S 1 , we have 992

<!-- formula-not-decoded -->

which causes a contradiction to the fact that P Ψ ( π ⋆ 1 , π ⋆ 2 ) = Ψ(1 / 2) . 993

̸

̸

Case 2. If supp( π ⋆ 1 ) ⋂ S 1 = ∅ , and supp( π ⋆ 1 ) ⋂ S c 1 = ∅ , taking ˜ π ⋆ 2 as: 994

<!-- formula-not-decoded -->

## F Proofs of Results in Section 5 999

## F.1 Proof of Lemma 5.1 1000

Proof of Lemma 5.1. Suppose each player has n policies and the payoff matrix is { α ij } n i =1 . Then,

<!-- formula-not-decoded -->

Let us reformulated it into a convex optimization problem. 1001

<!-- formula-not-decoded -->

Let us further reformulate this problem into the epigraph form by introducing a single variable t ∈ R : 1002

<!-- formula-not-decoded -->

By introducing the dual variables u ∗ ∈ R n , ˜ u ∗ ∈ R n and v ∗ ∈ R , the KKT conditions is: 1003

- stationary condition:
- primal feasibility:
- dual feasibility:

<!-- formula-not-decoded -->

- complementary slackness:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can easily see that Slater's condition is satisfied for this problem, so the KKT points are equivalent 1004 to primal and dual solutions. Then taking π ∗ &gt; 0 into account, we have ˜ u ∗ i = 0 by the second 1005 complementary slackness condition, and the above equations can be simplified to the following 1006 system of equations: 1007

<!-- formula-not-decoded -->

Moreover, notice that 1008

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

thus v = -t 1009

<!-- formula-not-decoded -->

F.2 Verifying Equation (5.2) and Equation (5.3) 1010

For (5.2), choosing t ∗ = -v ∗ = ∑ n i =1 ( π ∗ i ) 2 and u ∗ i = v ∗ i , π ∗ is a Nash solution. For (5.3), choosing 1011 t ∗ = -v ∗ = 0 and u ∗ j = ( π ∗ j ) -1 ∑ n j =1 ( π ∗ j ) -1 , π ∗ is a Nash solution. 1012

## F.3 Proof of Theorem 5.1 1013

We first present a useful lemma (Lemma F.1) that further investigates the KKT conditions (Lemma 1014 5.1) when the payoff matrix induces a unique Nash equilibrium. 1015

Lemma F.1. If a game with the payoff matrix { α ij } n i,j =1 has a unique Nash solution π ∗ , then for 1016 any j ∈ [ n ] , it must hold u ∗ j &gt; 0 in the KKT conditions, and 1017

<!-- formula-not-decoded -->

Proof of Lemma F.1. Suppose that the KKT conditions provide the unique Nash solution ( π ∗ , u ∗ , t ∗ ) . 1018 Then we define: 1019

̸

<!-- formula-not-decoded -->

̸

with J 0 ∪ ˜ J 0 = [ n ] . Since u ∗ ⩾ 0 and ∑ u ∗ j = 1 , there exists j ∈ [ n ] , such that u ∗ j = 0 , i.e., J 0 = ∅ . 1020 Now, we aim to show ˜ J 0 = ∅ . We prove by contradiction. Suppose ˜ J 0 = ∅ , taking j 0 ∈ J 0 , we 1021 consider two spaces 1022

<!-- formula-not-decoded -->

Then we claim that π ∗ ∈ V 2 and dim( V 2 ) ⩾ 2 . For the first claim, by the KKT conditions in Lemma 1023 5.1, for any j ∈ J 0 , we obtain 1024

<!-- formula-not-decoded -->

thus π ∗ ∈ V 1 . Moreover, again by the KKT conditions, for any j ∈ ˜ J 0 , we have 1025

<!-- formula-not-decoded -->

̸

̸

which shows that π ∗ ∈ V 2 . For the second claim, take ˜ j 0 ∈ ˜ J 0 and consider 1026

<!-- formula-not-decoded -->

We can easily see V 4 ⊆ V 2 . Note that V 3 can be regarded as a kernel space of a linear transformation from R n to R n -2 . By the dimension theorem in linear algebra, we obtain dim( V 3 ) = n -dim(Im( A )) ⩾ n -( n -2) = 2 . For any π ∈ V 3 , it must hold that π ∈ V 4 or -π ∈ V 4 , so dim( V 4 ) = dim( V 3 ) ⩾ 2 . Therefore, we have dim( V 1 ) ⩾ dim( V 2 ) ⩾ dim( V 4 ) ⩾ 2 .

̸

Thus, we can take another ˜ π ∗ ∈ V 2 which is linear independent with π ∗ . Note that for any a, b ∈ R + , a π ∗ + b ˜ π ∗ ∈ V 2 . Taking large a ∈ R + , we have a π ∗ + b ˜ π ∗ ∈ V 2 and a π ∗ + b ˜ π ∗ &gt; 0 , since π ∗ &gt; 0 . Therefore, there exists a 1 ∈ R + , such that π ∗ 2 := a π ∗ +˜ π ∗ a 1 ∈ V 2 that satisfies π ∗ 2 = π ∗ , π ∗ 2 &gt; 0 , and ∑ i π ∗ 2 ,i = 1 . Thus, we obtain another Nash equilibrium ( π ∗ 2 , u ∗ , t ∗ ) , causing contradiction to the uniqueness of Nash solution. Hence, it must hold that ˜ J 0 = ∅ .

Next we provide the proof for Theorem 5.1.

Proof of Theorem 5.1. Using Lemma F.1, uniqueness requires us to seek solutions that satisfies

<!-- formula-not-decoded -->

for all j ∈ [ n ] , where t ∗ is a constant that may depend on π ∗ . Consider n ⩾ 5 , for any four distinct 1037 indices j 1 , j 2 , k 1 , k 2 , we have 1038

̸

<!-- formula-not-decoded -->

̸

Let us consider the infinitesimal variation π k 1 → π k 1 + δ and π k 2 → π k 2 -δ , keeping others still. 1039 We obtain that 1040

̸

<!-- formula-not-decoded -->

̸

Subtracting both sides of (F.1) from (F.2), we obtain that 1041

<!-- formula-not-decoded -->

i.e., we have 1042

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

<!-- formula-not-decoded -->

As f is smooth, using 1043

1052

<!-- formula-not-decoded -->

and taking δ → 0 , we obtain the following identity from (F.4), 1044

<!-- formula-not-decoded -->

Thus, we obtain that 1045

<!-- formula-not-decoded -->

̸

Since (F.6) holds for any π &gt; 0 , given any π j 1 = π j 2 , for any x 1 , x 2 ∈ (0 , 1 -π j 1 -π j 2 ) , we have 1046

<!-- formula-not-decoded -->

which induces the following for any x ∈ (0 , 1 -π j 1 -π j 2 ) , 1047

<!-- formula-not-decoded -->

i.e., we have 1048

<!-- formula-not-decoded -->

Without any loss of generality, we assume π j 1 &lt; π j 2 , then we obtain 1049

<!-- formula-not-decoded -->

Taking limit, it must hold C ( π j 1 , π j 2 ) = 0 , i.e., we have 1050

<!-- formula-not-decoded -->

Since (F.9) holds for any π &gt; 0 , for any x 1 , x 2 ∈ R + , we have 1051

<!-- formula-not-decoded -->

thus, for any x ∈ R + , we have

<!-- formula-not-decoded -->

Solving (F.10), we obtain that

Then we obtain that 1053

<!-- formula-not-decoded -->

̸

yielding C 3 = C +( n -1) C 2 , and 1054

<!-- formula-not-decoded -->

which is contradictory to our assumptions. 1055

<!-- formula-not-decoded -->
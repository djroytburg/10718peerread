<!-- image -->

## Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models

Sreyan Ghosh ★ 12 , Arushi Goel ★ 1 , Jaehyeon Kim 1 , Sonal Kumar 2 , Zhifeng Kong 1 , Sang-gil Lee 1 , Chao-Han Huck Yang 1 , Ramani Duraiswami 2 , Dinesh Manocha 2 , Rafael Valle 1 , Bryan Catanzaro 1

NVIDIA, USA 1 , University of Maryland, College Park, USA 2

★ Equally contributed and led the project. Work done by Sreyan during an internship at NVIDIA.

Correspondence: arushig@nvidia.com, sreyang@umd.edu

Project: https://research.nvidia.com/labs/adlr/AF3/

## Abstract

We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multiaudio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closedsource models trained on much larger datasets.

## 1 Introduction

Audio-including speech, sounds, and music-is central to human perception and interaction. It enables us to understand our surroundings, engage in conversations, express emotions, interpret videos, and enjoy music. For AI systems to approach artificial general intelligence (AGI) [88], they must similarly develop the ability to comprehend and reason over diverse audio signals. While Large Language Models (LLMs) excel at language-based reasoning, their audio comprehension remains limited - both in accessibility and capability [54, 106]. Extending LLMs to process and reason over audio is essential for building truly context-aware, intelligent agents.

Audio-Language Models (ALMs) extend the capabilities of LMs to the auditory domain. Early works such as CLAP [33] align audio and text in a shared embedding space, enabling them with tasks like retrieval [89].

Figure 1: AF3 vs. prior SOTA LALMs (values normalized and WER=100-WER). AF3 outperforms most open-source/weights (e.g., Qwen2.5Omni) and closed (e.g., Gemini 2.5 Pro) LALMs while being fully open.

<!-- image -->

Table 1: Comparison of various LALMs in terms of capabilities and openness. AF3 stands out as the most capable and open model to date, achieving SOTA results across benchmarks (A in Chat stands for Audio). * Voice-out is powered by our novel streaming TTS implementation, which is also applicable to other LALMs.

<!-- image -->

| Models           | Audio Understanding   | Audio Understanding   | Audio Understanding   | Voice   | Voice   | Multi-turn Chat   | Multi-turn Chat   | Long Audio ( > 30 secs)   | Long Audio ( > 30 secs)   | Long Audio ( > 30 secs)   | Open-Source   | Open-Source   | Open-Source   |
|------------------|-----------------------|-----------------------|-----------------------|---------|---------|-------------------|-------------------|---------------------------|---------------------------|---------------------------|---------------|---------------|---------------|
| Models           | Sound                 | Music                 | Speech                | In      | Out*    | Single A          | Multiple A        | Speech                    | Sound                     | Music                     | Model         | Data          | Code          |
| LTU              | ✓                     | ✓                     | ×                     | ×       | ×       | ×                 | ×                 | ×                         | ×                         | ×                         | ✓             | ✓             | ✓             |
| LTU-AS           | ✓                     | ✓                     | ✓                     | ×       | ×       | ×                 | ×                 | ×                         | ×                         | ×                         | ✓             | ✓             | ✓             |
| GAMA             | ✓                     | ✓                     | ×                     | ×       | ×       | ×                 | ×                 | ×                         | ×                         | ×                         | ✓             | ✓             | ✓             |
| SALMONN          | ✓                     | ✓                     | ✓                     | ×       | ×       | ×                 | ×                 | ×                         | ×                         | ×                         | ✓             | ✓             | ✓             |
| MuLLaMa          | ×                     | ✓                     | ×                     | ×       | ×       | ×                 | ×                 | ×                         | ×                         | ×                         | ✓             | ✓             | ✓             |
| Phi-4-mm         | ✓                     | ✓                     | ✓                     | ×       | ×       | ×                 | ×                 | ✓                         | ✓                         | ✓                         | ✓             | ×             | ×             |
| Qwen-Audio       | ✓                     | ✓                     | ✓                     | ✓       | ✓       | ✓                 | ×                 | ×                         | ×                         | ×                         | ✓             | ×             | ×             |
| Qwen2-Audio      | ✓                     | ✓                     | ✓                     | ✓       | ✓       | ✓                 | ×                 | ×                         | ×                         | ×                         | ✓             | ×             | ×             |
| Qwen2.5-Omni     | ✓                     | ✓                     | ✓                     | ✓       | ✓       | ✓                 | ×                 | ✓                         | ✓                         | ✓                         | ✓             | ×             | ×             |
| GPT-4o Audio     | ✓                     | ✓                     | ✓                     | ✓       | ✓       | ✓                 | ✓                 | ✓                         | ✓                         | ✓                         | ×             | ×             | ×             |
| Gemini 2.0 / 2.5 | ✓                     | ✓                     | ✓                     | ✓       | ✓       | ✓                 | ×                 | ✓                         | ✓                         | ✓                         | ×             | ×             | ×             |
| Audio Flamingo   | ✓                     | ✓                     | ×                     | ×       | ×       | ✓                 | ×                 | ×                         | ×                         | ×                         | ✓             | ✓             | ✓             |
| Audio Flamingo 2 | ✓                     | ✓                     | ×                     | ×       | ×       | ×                 | ×                 | ×                         | ✓                         | ✓                         | ✓             | ✓             | ✓             |
| Audio Flamingo 3 | ✓                     | ✓                     | ✓                     | ✓       | ✓       | ✓                 | ✓                 | ✓                         | ✓                         | ✓                         | ✓             | ✓             | ✓             |

More recently, the emergence of Large ALMs (LALMs)-decoder-only language models augmented with audio understanding [20, 19, 105]-has unlocked powerful capabilities, including open-ended audio question-answering (AQA) that demands both reasoning and world knowledge [101]. These capabilities have further enabled tasks like audio analysis [32, 60], conversational assistants [24], etc.

However, existing models still fall short in key areas critical to AGI, such as expert-level reasoning [88, 101], multi-turn and multi-audio dialogue [44], and long audio understanding [40]. We identify two core limitations: (i) most LALMs are trained primarily on short audio for recognition tasks rather than ones that require deliberate reasoning; and (ii) in turn, they lack exposure to the skill sets required for complex tasks. Additionally, most LALMs that support all three modalities of speech, sound, and music are closed-source: while some have publicly released models weights [20, 19, 1], they offer limited to no information about their data, code, or recipes (more details in Table 1).

Main Contributions. To address these issues, we introduce Audio Flamingo 3 (AF3) , a fully open-source 1 LALM with state-of-the-art performance in audio understanding and reasoning across 20+ benchmarks. In addition, AF3 brings several novel capabilities, including multi-turn, multi-audio chat, on-demand thinking, voice-to-voice interaction, and long-context audio reasoning (up to 10 minutes). We propose three core innovations to enable these capabilities: (i) Data : We focus on curating high-quality data at scale and propose (a) AudioSkills-XL : a large-scale dataset of 8M diverse AQA pairs, (b) LongAudio-XL : large-scale dataset of 1.25M diverse audio QA pairs for long audio reasoning; (c) AF-Chat : a multi-turn multi-audio chat dataset curated using a novel algorithm with 75k instances and (d) AF-Think : a dataset with 250k+ AQA pairs with short length prefixes to encourage CoT-type reasoning before arriving at the answer (ii) AF-Whisper : We train AF-Whisper, a unified audio encoder pretrained using a novel strategy on large-scale audio-caption pairs, capable of learning general-purpose representations across speech, sounds, and music; and (iii) Learning Curriculum : We train AF3 with a five-stage curriculum-based training strategy that progressively increases context length and task complexity. In summary, our main contributions are:

- We introduce Audio Flamingo 3 (AF3) , the most open and capable foundational LALM to date. AF3 introduces key capabilities including: (i) long-context audio QA (extending beyond sounds as in [40] and including speech), and (ii) flexible, on-demand thinking, enabling the model to generate concise, CoT-style reasoning steps when prompted. AF3 achieves state-of-the-art performance on 20+ audio understanding and reasoning benchmarks.
- We also present AF3-Chat , a fine-tuned variant of AF3 designed for multi-turn, multi-audio chat and voice-to-voice interaction.
- We propose novelties in data curation, audio encoder representation learning, and training strategies. Being fully open, we release our code, training recipes, and 4 new datasets to promote research in this space.

1 By fully open , we mean that the model's weights, training data, and code will be publicly released, with full transparency about the training methodology. Due to the licensing and scope of the training data used in the work, all releases will be under a research-only license.

## 2 Related Work

Audio Language Models. The rapid progress of LLMs has catalyzed the development of multimodal LLMs (MLLMs) capable of understanding and reasoning across diverse data modalities, including audio. Within this space, ALMs specifically target reasoning over auditory inputs such as speech, sounds, and music. ALMs typically follow two main architectural paradigms: (i) Encoder-only ALMs , which learn a joint embedding space for audio and text, enabling tasks like cross-modal retrieval. Representative models include CLAP [33], Wav2CLIP [112], and AudioCLIP [48]. (ii) Encoderdecoder ALMs , also referred to as LALMs, which use decoder-only LLMs augmented with an audio encoder. Notable examples include LTU [46], LTU-AS [45], SALMONN [104], Pengi [27], Audio Flamingo [65], Audio Flamingo 2 [40], AudioGPT [53], GAMA [41], Qwen-Audio [20], and Qwen2Audio [19]. These LALMs have significantly improved performance on core audio understanding tasks such as automatic speech recognition (ASR) [96], audio captioning [60], and acoustic scene classification [16]. More importantly, they have enabled new capabilities such as open-ended audio question answering, which requires complex reasoning and external world knowledge.

Despite these advancements, current LALMs fall short in supporting various capabilities, including multi-turn, multi-audio chat, long-context audio comprehension, etc. Moreover, most LALMs are limited to specific audio types, lacking the ability to unify understanding across speech, sounds, and music. Finally, the most advanced LALMs remain only partially open, releasing model checkpoints without accompanying training code or data. This lack of transparency limits reproducibility and impedes scientific progress by obscuring the development process.

Reasoning and Long-Context Understanding. Recent progress in LLMs has increasingly emphasized long-context understanding. In the vision-language space, substantial strides have been made in modeling long videos [17]. In the audio domain, AF2 marked the first step toward long-context audio comprehension, though it is limited to sounds and music.

Parallel efforts have aimed to enhance reasoning in LLMs and MLLMs through improved reasoning datasets [101, 110], advancements in multimodal perception [116, 105], and emerging paradigms like chain-of-thought (CoT) prompting [80], which encourages models to "think before answering." In developing AF3, we combine these advances-integrating controlled reasoning supervision, longcontext training, and modality diversity-to equip the model with strong reasoning capabilities and long-context comprehension, including speech.

## 3 Methodology

## 3.1 Audio Flamingo 3 Architecture

In this section, we discuss our proposed architecture for Audio Flamingo 3 as shown in Figure 2. AF3 consists of i) AF-Whisper: an audio encoder with sliding window feature extraction, ii) audio projector, iii) an LLM, and iv) a streaming TTS. We provide details of each component below.

AF-Whisper Audio Encoder. Prior work in audio representation learning typically treats speech, sounds, and music as separate modalities, and LALMs often rely on distinct encoders for each [104, 41]. Using separate encoders for LALMs increases model complexity, introduces framerate mismatches, and can lead to training instability. To address this, we propose AF-Whisper, a unified audio encoder trained with a simple yet effective representation learning strategy to model all three audio types.

As illustrated in Figure 2, we start with the pre-trained Whisper large-v3 encoder [96], attach it to a standard Transformer decoder, and train using the audio captioning task with the next-token-prediction objective. To achieve this, we generate a natural language caption for each audio, describing its speech, sound, and music content. First, we pool several datasets and then prompt GPT-4.1 to generate the audio caption. For prompting, we use available metadata for each sample, which includes transcripts, ambient sound descriptions, and music attributes. For samples lacking any of the 3 metadata, we synthesize it using AF2 [40] or Whisper-Large-v3 ASR [96]. All datasets used for training are detailed in Section A.2. We choose Whisper as the backbone due to its existing speech understanding capabilities and its dense, high-resolution audio features, which are more informative than those from models like CLAP [33]. We connect it with a Transformer decoder using cross-attention (similar to RECAP[77] and AF2 [40]) with 24 layers, 8 attention heads, and 1024 hidden size.

Figure 2: Overview of Audio Flamingo 3, AF-Whisper training, and five-stage curriculum training.

<!-- image -->

Feature Extraction. Given an audio input A , we first resample it to 16kHz mono. The raw waveform is then transformed into a 128-channel mel-spectrogram using a window size of 25ms and a hop size of 10ms. This mel-spectrogram is processed by AF-Whisper, producing hidden representations, denoted as h a = f a ( A ) , where h a ∈ R N × d . As shown in Figure 2, each audio is processed in 30-second chunks of non-overlapping sliding windows, and N or the temporal resolution depends on the length of the audio and the maximum number of sliding windows (which varies according to the stage of training). AF-Whisper produces audio features at a frame rate of 50Hz, and we further apply a pooling layer with a stride of two similar to [19]. d denotes the hidden dimension, which is 1280.

Audio Adaptor. To align the audio modality with the text embedding space of the LLM, we introduce audio adaptor layers, denoted by A ( . ) . Specifically, the encoded hidden representations h a from AF-Whisper are passed through these adaptor layers (2-layer MLP with a GeLU layer) to produce embeddings: a = A ( h a ) . These resulting embeddings serve as prompts to the LLM, alongside the textual instruction.

Large Language Model (LLM). We employ Qwen-2.5-7B [118] as our backbone, a decoder-only causal LLM with 7B parameters, 36 hidden layers, and 16 attention heads.

Streaming TTS. To enable voice-to-voice interaction, we employ a TTS module for streaming speech generation, supporting streaming inputs and outputs. Our TTS module employs a decoderonly transformer architecture: it predicts the subsequent audio token conditioned on incoming subword text tokens from the LLM and the history of previously generated audio tokens. Similar streaming TTS techniques have been explored with LLMs [115] (for voice-out on LLM outputs), but not in the context of LALMs (which we define as models designed to perceive and reason over diverse audio inputs). Since not a core novelty of our work, we provide more details, including training and architecture, in Appendix I.

## 4 Audio Flamingo 3 Training Data

We present detailed statistics for all datasets used to train AF3 in Table 11. AF3 has a total of 5 stages of training, where each stage employs a unique combination of datasets with unique weights (number of passes over that dataset for that particular stage). For Stages 1 and 2, we use open-source, recognition-focused foundational datasets converted to QA format. In the following sub-sections, we introduce our four novel skill-focused and unique datasets, each accompanied by custom data curation strategies, used in Stages 3, 3.5, and 4, which form a core contribution of this work.

## 4.1 AudioSkills-XL: Expanding AudioSkills with Reasoning-Focused QAs

Audio QA pairs derived from foundational benchmarks focused on recognition tasks (e.g., ASR, acoustic event classification) are insufficient for training models in expert-level reasoning [101]. Therefore, in Stage 3 fine-tuning, we prioritize the development of reasoning and problem-solving abilities by curating large-scale, high-quality Audio QA data. Inspired by AF2, we limit this stage to short audio clips ( ≤ 30s) and defer long audio reasoning to later stages. We expand the AudioSkills

dataset [40] by 4.5M new Audio QA pairs (majorly multiple-choice questions (MCQ)-based) to create AudioSkills-XL, a high-quality corpus containing 8M Audio QA pairs, using two strategies:

(1) We expand coverage of existing reasoning skills and introduce new ones using additional audio sources, increasing the dataset by 3.5M QA pairs : (a) For sounds, we incorporate data from YouTube8M and synthetic sources. (b) For music, we include Music4All [102] and the Million Song Dataset [8]. For YouTube8M, we adapt captions from AudioSetCaps [6] and generate QA using GPT-4.1 with general reasoning prompts from AF2. Additionally, we introduce new reasoning skills and design corresponding prompts to support them. For music, we generate data for novel skills (as AudioSkills was focused more on sounds; details in Table 6) and go beyond captions we leverage metadata such as song titles, artist names, album names, etc (see Fig. 4 for full list) to generate more complex, reasoning-focused QAs. We also use this metadata to generate rich music captions for Stage 1 and 2 pre-training (see Fig. 4), demonstrating how text-based knowledge can enhance audio understanding, particularly in knowledge-driven domains like music. This method can be seen as synthetic knowledge generation, where we leverage text-based knowledge to enrich audio understanding and enable models to acquire domain-specific knowledge from unlabeled audios in the wild. Our analysis shows that LLMs like GPT-4.1 hold substantial world knowledge about music, and that metadata improves QA quality significantly.

(2) We augment AudioSkills with 1M speech QA samples using YouTube8M [2], LibriSpeech [92] (read speech), GigaSpeech [14] (conversational), and VoxCeleb2 [21] (interviews). From YouTube8M, we introduce a new task: Speech-in-Sound QA, where the model must reason over both speech content and ambient sounds to understand complex auditory scenes. To create these QAs, we create Speech-in-Sound-Caps, a new dataset with ≈ 2M speech-aware auditory scene captions from YouTube8M. To curate this, we first filter the dataset for English speech (using AF2) and transcribe the spoken content with Whisper-Large-v3. We then generate two types of descriptions: one capturing sound events and another summarizing speech characteristics such as tone, emotion, and pitch (both using AF2 and custom prompts; see Appendix 26). Finally, we prompt GPT-4.1 to synthesize a speech-aware scene caption. These captions significantly improve the quality of final audio captions (compared to only using sound information) by providing a more holistic representation of the audio. For LibriSpeech and GigaSpeech, we concatenate shorter segments into clips of 15-30 seconds, selecting information-dense segments filtered by prompting an LLM. To move beyond basic spoken content understanding common in most current datasets [121], we design five distinct types of speech QA that require diverse reasoning skills (explained in the next subsection).

## 4.2 LongAudio-XL: Expanding LongAudio with Long Speech QA

To our knowledge, Long Speech QA (i.e., audio ≥ 30 seconds) has not been explored in prior work, despite its relevance to real-world applications such as long-form conversation understanding, meeting summarization, and narrative comprehension. To bridge this gap, we extend the existing LongAudio dataset [40] (focused on sounds and music) by incorporating over 1M reasoning-focused QA examples from long-form speech (30s-10min). We curate audios from diverse sources including: Single-speaker speech : LibriSpeech (audiobooks) [92], EuroParl [62], VoxPopuli (parliamentary debates) [107] and Multi-speaker conversations : Spotify Podcasts [23], Switchboard [43], Fisher (dyadic calls) [22], MELD [94], DailyTalk [71], MMDialog (natural dialogues) [35]. We merge consecutive short segments in chronological order to construct longer, coherent audios. We construct QAs across a wide range of skills, as illustrated in Figure 3:

1. Sarcasm Identification: Inferring sarcasm by analyzing content, tone, and emotional cues.
2. Emotional State Reasoning: i) Identification: Determine the speaker's emotion at a specific utterance. ii) Causal Reasoning: Identify the reason behind a speaker's emotional state using conversational context. iii) Emotion Flip: Explain shifts in a speaker's emotional state during the conversation.
3. Topic Relationship Reasoning: Understand how two ideas or topics are related within the overall discourse.
4. Information Extraction (IE): i) Needle QA: Targeted QA on specific utterances or parts of the speech (e.g., entity or fact extraction, general knowledge linkage). ii) Causal QA: Identify causes for a particular utterance in context. iii) Response QA: Extract how one speaker responds to another's statement. iv) Topic QA: Identify the main topic of the speech or conversation.
5. Summarization: Generate a concise summary of the speech content.

Figure 3: Examples from AudioSkill-XL , LongAudio-XL , AF-Think , and AF-Chat. We include additional examples in Appendix B and C, featuring novel music reasoning QAs mentioned in detail in Section B.1.2.

<!-- image -->

6. Order: i) Temporal Order: Understanding the sequential order of topics in the speech; ii) Temporal Attribute: Understanding how topics change over time; iii) Temporal Referring: Resolve references to specific time points (e.g., "at the end") iv) Temporal Grounding: Identify when in the audio a specific topic was discussed.

## 4.3 AF-Think: Towards flexible, on-demand reasoning

Recent studies show that making an LLM 'think', similar to chain-of-thought (CoT) prompting [111], can improve reasoning performance in LLMs [47], especially for complex tasks like coding and math (e.g., DeepSeek-R1, OpenAI-o1). Visual MLLMs have also benefited from this paradigm [116, 109]. In the audio domain, early attempts such as Audio-CoT [80], Audio-Reasoner [114], and R1AQA [73] have explored CoT-style reasoning, but often yield limited gains and involve complex or inefficient training procedures. Moreover, consistent with findings in [73], we observe that deep, explicit thinking does not always improve performance in audio understanding tasks.

In AF3, we adopt a lightweight thinking mechanism with two key modifications: (i) We create AF-Think, a dataset of 250k MCQ-based QAs with short, controlled thought preceding the answer. This additional thinking serve as a prefix to the answer and are limited to an average of approximately 40 words, providing concise yet effective context for audio QA (example in Figure 3). (ii) Instead of explicitly post-training for CoT, we add a special suffix to QA prompts (highlighted in Figure 3). We include AF-Think in the Stage 3.5 training mixture, upweighted relative to standard QA data. This allows AF3 to think only when prompted, offering flexible, on-demand additional reasoning .

To generate AF-Think, we first sample a subset of multiple-choice reasoning QAs from AudioSkillsXL and LongAudio-XL (originally with just the correct option as the answer). Next, we prompt Gemini 2.0 Flash with the input audio, the question, and the answer to generate short thinking prefixes. We found Gemini to hallucinate less and generate more accurate reasoning when guided by the ground-truth answer, rather than producing CoT from scratch. We restrict this process to only high-quality datasets and filter out noisy instances.

## 4.4 AF-Chat: Multi-turn Multi-audio Chat Data

While single-turn single-audio QA training equips LALMs to reason over individual audio inputs, enabling free-form, multi-turn, multi-audio conversations requires a dedicated chat alignment tuning stage, akin to the instruction-tuning phases used for LLMs [122]. Chat becomes significantly more complex when multiple audio inputs must be integrated across turns, requiring the model to track context, reason over relationships between past and current inputs, and generate coherent follow-ups. Despite its importance and chat being the most used application of LLMs, this capability remains underexplored in LALMs primarily due to the absence of open, high-quality training data.

To address this gap, we introduce AF-Chat , a high-quality fine-tuning dataset consisting of 75k multi-turn, multi-audio chat instances. On average, each dialogue includes 4.6 audio clips and 6.2 dialogue turns, with a range of 2-8 audio clips and 2-10 turns. To construct this dataset, we draw

from Speech-in-Sound Caps (for speech and sounds), and Music4All and MSD (for music). We follow a two-step curation process: First, for each seed audio, we identify its top 8 most semantically similar and dissimilar clips using a combination of captions, NV-Embed-v2 [68] embeddings, and FAISS-based clustering [31] (details in Appendix E.2). For every dialogue, we restrict the audios to this pool. This targeted clustering yields significantly higher-quality dialogues than random audio selection by ensuring each instance is grounded in a diverse yet semantically coherent audio pool.

Next, we prompt GPT-4.1 using carefully designed expert exemplars (Fig. 36 and 35) to generate natural, multi-turn chat sessions under the following constraints: (i) the model may choose any subset of the similar/dissimilar audios (up to 10 turns), prioritizing conversation quality; (ii) not all turns require a new audio-follow-up and clarification questions are encouraged; and (iii) later turns may refer back to earlier audios or responses to simulate real conversational grounding. The design of AF-Chat is informed by extensive internal human studies to reflect how users naturally interact with audio-language models. As a result, it provides rich, diverse supervision for aligning LALMs to handle complex, contextual, and naturalistic audio conversations. Finally, we select 200 high-quality samples for the test set, known as AF-Chat-test, and ensure that the audios in these instances have audio clips that were not seen during training.

## 5 Audio Flamingo 3 Training Strategy

AF3 is trained using a five-stage strategy designed to progressively enhance its capabilities by increasing audio context length, improving data quality, and diversifying tasks. A full list of datasets used at each stage is provided in Appendix 11.

Stage 1: Alignment pre-training. For this stage, we train only the audio adaptor layers while keeping the audio encoder and LLM frozen. This step aligns encoder representations with the language model. Stage 2: Encoder Tuning. The main purpose of this stage is to adapt AF-Whisper to diverse datasets and broaden and improve its audio understanding capabilities. We fine-tune both the audio encoder and adaptor while keeping the LLM frozen. In both Stages 1 and 2, the audio context length is limited to 30 seconds, and training uses recognition-focused datasets (e.g., classification, captioning, and ASR). Stage 3: Full Fine-Tuning. The primary purpose of this stage is to emphasize reasoning and skill acquisition by the LALM. As mentioned earlier, since skill-specific data is easy to scale on short audios, we still stick to short audios in this stage and use high-quality foundational and QA datasets and our proposed AudioSkills-XL. However, we increase the audio context length up to 2.5 minutes now to accommodate the moderately long audios in AudioSkills. The resulting model at the end of Stage 3.5 is referred to as AF3. Stage 3.5: Context Extension and Thinking. This stage focuses on extending context length and encouraging CoT-style reasoning. In addition to the Stage 3 data mixture, we incorporate LongAudio-XL and AF-Think. We adopt LoRA-based training [51]-similar to LTU and GAMA-by freezing the model's original weights and training LoRA adapters for the LLM. This approach allows end-users to flexibly enhance the model's reasoning and long-context understanding capabilities on demand. Stage 4: Chat and Voice Fine-Tuning. This stage focuses on enabling multi-turn, interactive, and voice-based dialogue. We fine-tune the entire model on our proposed AF-Chat dataset to equip AF3 with conversational audio understanding and response generation capabilities. The resulting model at the end of Stage 4 is referred to as AF3-Chat.

## 6 Experiments

Experimental Setup. We train AF3 on 128 NVIDIA A100 GPUs, each with 80GB of memory. Details about batch size, learning rates, and optimizers for each stage of training are in Appendix H.

Baselines. We evaluate our model against recent SOTA LALMs, including GAMA [41], Audio Flamingo [65], Audio Flamingo 2 [40], Qwen-A(udio) [20], Qwen2-A(udio) [19], Qwen2-A(udio)(Inst)ruct, Qwen2.5-O(mni) [117], R1-AQA [73], Pengi [27], Phi-4-mm [1], Baichun Audio [75], Step-Audio-Chat [52], LTU [46], LTU-AS [45], SALMONN [104], AudioGPT [53], and Gemini (2.0 Flash, 1.5 Pro, 2.5 Flash and 2.5 Pro) [105] (note we do not evaluate Gemini on ASR benchmarks due to low rate limits), as well as GPT-4o-audio [54]. For LongAudioBench, for models that do not support longer audio, we follow the cascaded approach for evaluation proposed by [40]. For Table 3, we only compare against open LALMs. All results reported in the tables correspond to the best-performing model. Evaluation for voice-to-voice capabilities is beyond our scope.

Table 2: Comparison of AF3 with other LALMs on various benchmarks (WER ↓ (Word Error Rate), ACC ↑ (Accuracy), and GPT4o ↑ (GPT evaluation)). We report scores for only the top-performing prior LALM. +Think refers to AF3 with additional thinking. We highlight closed source, open weights, and open source models.

| Task                               | Dataset                                                                | Prior SOTA                                                | Metrics   | Results                                                                                                                                |
|------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------|
| Audio and Reasoning                | MMAU-v05.15.25 (test) Sound &#124; Music &#124; Speech &#124; Avg      | Qwen2.5-O Audio Flamingo 3 +Think                         | ACC ↑     | 76.77 &#124; 67.33 &#124; 68.90 &#124; 71.00 75.83 &#124; 74.47 &#124; 66.97 &#124; 72.42 75.27 &#124; 74.60 &#124; 69.60 &#124; 73.16 |
| Audio and Reasoning                | MMAU-v05.15.25 (test-mini) Sound &#124; Music &#124; Speech &#124; Avg | Qwen2.5-O Audio Flamingo 3 +Think                         | ACC ↑     | 78.10 &#124; 65.90 &#124; 70.60 &#124; 71.50 79.58 &#124; 73.95 &#124; 66.37 &#124; 73.30 79.88 &#124; 76.55 &#124; 66.37 &#124; 74.26 |
| Audio and Reasoning                | MMAR                                                                   | Qwen2.5-O Audio Flamingo 3 +Think                         | ACC ↑     | 56.7 58.5 60.1                                                                                                                         |
| Audio and Reasoning                | MMSU                                                                   | Gemini-1.5-Pro Audio Flamingo 3 +Think                    | ACC ↑     | 60.7 61.4 62.3                                                                                                                         |
| Audio and Reasoning                | ClothoAQA unanimous &#124; non-binary                                  | Qwen2.5-O &#124; Qwen2.5-O Audio Flamingo 3               | ACC ↑     | 89.2 &#124; 52.6 91.1 &#124; 56.2                                                                                                      |
| Audio and Reasoning                | Audio Captioning Clotho-v2 &#124; AudioCaps                            | Audio Flamingo 2 &#124; Audio Flamingo 2 Audio Flamingo 3 | CIDEr ↑   | 0.46 &#124; 0.58 0.50 &#124; 0.70                                                                                                      |
| Understanding                      | Audio Entailment Clotho &#124; AudioCaps                               | Audio Flamingo 2 &#124; Audio Flamingo 2 Audio Flamingo 3 | ACC ↑     | 92.5 &#124; 93.3 93.3 &#124; 95.0                                                                                                      |
| Understanding                      | IEMOCAP                                                                | Qwen2-A-Inst Audio Flamingo 3                             | ACC ↑     | 59.2 63.8                                                                                                                              |
| Understanding                      | CochlScene                                                             | Pengi Audio Flamingo 3                                    | ACC ↑     | 91.6 93.2                                                                                                                              |
| Understanding                      | NonSpeech7k                                                            | Audio Flamingo 2 Audio Flamingo 3                         | ACC ↑     | 84.3 85.9                                                                                                                              |
| Understanding                      | CMMHallucination                                                       | Gemini 2.5 Pro Audio Flamingo 3                           | ACC ↑     | 82.0 86.5                                                                                                                              |
| Understanding                      | CompA-R- test                                                          | Audio Flamingo 2 Audio Flamingo 3                         | ACC ↑     | 96.4 98.0                                                                                                                              |
| Understanding                      | MusicAVQA                                                              | Qwen2.5-O Audio Flamingo 3                                | ACC ↑     | 73.4 76.7                                                                                                                              |
| Understanding                      | NSynth Source &#124; Instrument                                        | Pengi &#124; Qwen-A Audio Flamingo 3                      | ACC ↑     | 62.0 &#124; 78.8 65.5 &#124; 78.9                                                                                                      |
| Understanding                      | Music Instruct Long                                                    | Audio Flamingo 2 Audio Flamingo 3                         | ACC ↑     | 90.2 92.7                                                                                                                              |
| Understanding                      | MuchoMusic                                                             | Qwen2-A-Inst Audio Flamingo 3 +Think                      | ACC ↑     | 46.2 47.4 47.6                                                                                                                         |
| Understanding                      | LibriSQA                                                               | Gemini 2.5 Pro Audio Flamingo 3                           | GPT4o ↑   | 8.7 8.7                                                                                                                                |
| Understanding                      | LongAudioBench                                                         | Gemini 2.5 Pro Audio Flamingo 3                           | GPT4o ↑   | 60.4 68.6                                                                                                                              |
| Understanding                      | +Speech (ours)                                                         | Gemini 2.5 Pro Audio Flamingo 3                           | GPT4o ↑   | 66.2 72.9                                                                                                                              |
| Automatic Speech Recognition (ASR) | LibriSpeech (en) test-clean &#124; test-other                          | Phi-4-mm &#124; Qwen2.5-O Audio Flamingo 3                | WER ↓     | 1.67 &#124; 3.4 1.57 &#124; 3.13                                                                                                       |
| Automatic Speech Recognition (ASR) | SPGISpeech (en)                                                        | Qwen2-A-Inst Audio Flamingo 3                             | WER ↓     | 3.0 1.86                                                                                                                               |
| Automatic Speech Recognition (ASR) | TEDLIUM (en)                                                           | Phi-4-mm Audio Flamingo 3                                 | WER ↓     | 2.9 3.5                                                                                                                                |
| Automatic Speech Recognition (ASR) | GigaSpeech (en)                                                        | Phi-4-mm Audio Flamingo 3                                 | WER ↓     | 9.78 10.27                                                                                                                             |
| Automatic Speech Recognition (ASR) | Common Voice 15 (en)                                                   | Phi-4-mm Audio Flamingo 3                                 | WER ↓     | 7.61 7.4                                                                                                                               |
| Automatic Speech Recognition (ASR) | VoxPopuli (en)                                                         | Phi-4-mm Audio Flamingo 3                                 | WER ↓     | 5.91 5.55                                                                                                                              |

Evaluation Datasets. We evaluate AF3 on a variety of tasks and benchmarks, including audio classification (CochlScene [57], NSynth (Source and Instrument) [34], NonSpeech7k [99], IEMOCAP [11]), audio QA (ClothoAQA [76], MusicAVQA [74], Music Instruct [26], LibriSQA [121]), reasoningfocused audio QA (MMAU [101] (v05.15.25), MuchoMusic (perceptual version) [120, 110], MMAR[81], MMSU [108], CompA-R-test [42], Audio Entailment [29]), multimodal hallucination

Table 3: Comparison of AF3 with open LALMs on AF-Chat, voice-text and TTS benchmarks. WER ↓ (Word Error Rate), SIM ↑ (Similarity), Human ↑ (Human evaluation) and GPT4o ↑ (GPT evaluation) indicate metrics and whether lower or higher is better.

| Task              | Dataset                                                            | Model                           | Metrics                          | Results                                                                                                                          |
|-------------------|--------------------------------------------------------------------|---------------------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Multi-audio chat  | AF-Chat-test Factuality &#124; Usefulness &#124; Depth             | Qwen2.5-O AF3-Chat              | Human ↑                          | 2.4 &#124; 2.7 &#124; 3.2 3.6 &#124; 3.4 &#124; 3.9                                                                              |
| Voice-Text        | OpenAudioBench alpaca-eval &#124; llama-questions &#124; trivia-qa | Qwen2-A-Inst Qwen2.5-O AF3-Chat | GPT4o ↑                          | 57.19 &#124; 69.67 &#124; 40.30 72.76 &#124; 75.33 &#124; 57.06 76.26 &#124; 80.33 &#124; 53.05                                  |
| Voice-Text        | VoiceBench AlpacaEval &#124; AdvBench &#124; OpenBookQA &#124;     | Qwen2-A-Inst Qwen2.5-O AF3-Chat | GPT4o ↑                          | 3.69 &#124; 98.85 &#124; 49.01 &#124; 3.40 4.33 &#124; 99.62 &#124; 79.12 &#124; 3.84 4.19 &#124; 98.26 &#124; 66.81 &#124; 3.40 |
| Voice-Text        | Commoneval                                                         |                                 |                                  |                                                                                                                                  |
| Speech Generation | SEED (test-en) Content Cons. &#124; Speaker Sim. &#124; Inf. Time  | Qwen2.5-O AF3-Chat              | WER ↓ &#124; SIM ↑ &#124; Time ↓ | 2.72 &#124; 0.63 &#124; 14.62s (1.26s) 2.02 &#124; 0.61 &#124; 5.94s (0.02s)                                                     |

detection (CMM [72]), audio captioning (Clotho-v2 [32], AudioCaps [60]), ASR (Librispeech (clean and other) [92], SPGISpeech [90], TEDLIUM [100, 49], GigaSpeech (Large) [14], Common Voice 15 [5] and Voxpopuli [107]) and long audio captioning and QA (LongAudioBench - which we augment with 2.5k human-annotated long-speech QA instances). For evaluating chat capabilities, we conduct a human study of model outputs on AF-Chat-test (more details in Appendix E) and compare only with Qwen2-Audio. Each annotator is asked to rate the response of the model for every turn on a scale of 1-5 for factuality, usefulness, and depth. We report results averaged across all instances across all turns. Furthermore, we evaluate the voice-text capabilities of our AF3-Chat model on two datasets, OpenAudioBench [75] and VoiceBench [18]. These benchmarks consist of voice queries (synthetically generated speech from text queries) and assess aspects such as instruction following, question answering, trivia knowledge, and reasoning. Finally, we evaluate our speech generation module using zero-shot TTS evaluation on the English subset of the SEED benchmark [4]. To calculate accuracy, we use either exact string matching with the ground truth or CLAP-based retrieval following [27], implemented with open-source AF-CLAP [40]. For MCQ, AF3 typically outputs only the selected option. In cases where the model provides more verbose or open-ended responses (e.g., with thinking mode), we apply multiple regex patterns to extract the chosen option.

## 6.1 Audio Understanding and Reasoning Evaluation

AF3 is the strongest and fully open-source LALM. Table 2 shows AF3 outperforming previous SOTA open-weight and closed-source models across a wide range of audio understanding and reasoning benchmarks. AF3 sets new highs on MMAU (72.42) (note for Qwen2.5-Omni on MMAU we report the 'parsed score' for fair evaluation), ClothoAQA (91.1), Clotho Entailment (92.9), and CMM Hallucination (86.7). On tasks like NSynth and MusicInstruct, it shows significant gains, highlighting strong sound and music understanding. For LongAudioBench (sound and speech), AF3 outperforms Gemini 2.5 Pro by a wide margin, demonstrating its strength in long-context reasoning. We also evaluate AF3 with thinking prompts (+Think) on reasoning-heavy benchmarks like MMAU and MuchoMusic, observing a performance boost. Although the thinking mode is activated after Stage 3.5 only when using our specific thinking prompt, the checkpoint remains usable without it. We report average scores of 73.16 and 74.26 on MMAU-test and MMAU-test-mini, respectively. Additionally, AF3 achieves state-of-the-art ASR results on LibriSpeech, SPGISpeech, and VoxPopuli-even compared to dedicated ASR models-despite not being trained on large-scale ASR datasets like many open-weight models. We illustrate a demo of AF3's capabilities in Fig. 14.

## 6.2 Chat and TTS Evaluation

Multi-turn multi-audio chat evaluation. On AF-Chat-test AF3-Chat shows a relative improvement of 30% over Qwen2.5-Omni, thereby showing the capability of effectively handling extended dialog turns, allowing for deeper contextual reasoning and more accurate references to multiple audio inputs.

Voice-Text and Speech Generation Evaluation. Table 3 evaluates AF3-Chat on two key tasks: voice-to-text and text-to-speech generation. In the voice-to-text setting (spoken QA), AF3-Chat achieves strong gains across all of OpenAudioBench, surpassing Qwen2.5-Omni. On VoiceBench, which tests spoken QA robustness across AdvBench, CommonEval, and OpenBookQA, AF3-Chat

performs comparably to Qwen2.5-Omni and Qwen2-Audio Chat. For TTS (evaluated on SEED testen), AF3-Chat shows improved performance with a lower WER of 2.02 (vs. 2.72 for Qwen2.5-Omni) and a speaker similarity score of 0.61, closely matching Qwen2.5's 0.63.

Furthermore, AF3-Chat exhibits significant advantages in generation speed. For a 10-second audio generation on an A100 GPU, AF3-Chat's text-to-audio token generation is 5.94 seconds with an additional 0.02 seconds for waveform synthesis. In comparison, the Talker model of Qwen2.5Omni requires 14.62 seconds for token generation and an additional 1.26 seconds for waveform synthesis. This efficiency allows our streaming text-to-speech to achieve a time-to-first-token of 0.15 seconds and an inter-token latency of 0.06 seconds (both including waveform synthesis), producing a 10-second audio clip in 6.68 seconds.

## 6.3 Ablation Studies

In this section, we ablate our key components (using just 10% of the training data) to support the paper's main claims.

Evaluating AF-Whisper as a Unified Encoder. Table 4 compares AF3 trained with our unified AF-Whisper encoder against a dual-encoder setup using CLAP for sounds/music and Whisper-v3 for speech [33, 96]. AF-Whisper outperforms the dual-encoder model under the same data budget, demonstrating its effectiveness as a single encoder for sound, music, and speech.

AudioSkills-XL: A Key Dataset for Performance Gains. : To measure the impact of AudioSkills-XL, we ablate it from Stage 3 of training and compare results to the full setup. As shown in Table 4, removing AudioSkills-XL causes a significant performance drop-particularly on MMAU-underscoring its role in improving generalization and robustness. These findings highlight the value of large-scale, skill-targeted audio QA data for fine-tuning multi-modal models.

Table 4: Comparison of AF3 w/ 10% data, w/o AF-Whisper and w/o AudioSkills-XL.

| Model              |   MMAU-Sound ACC ↑ |   MMAU-Music ACC ↑ |   MMAU-Speech ACC ↑ |   Librispeech-clean WER ↓ |   Librispeech-other WER ↓ |
|--------------------|--------------------|--------------------|---------------------|---------------------------|---------------------------|
| w/ 10% data + w/o  |               66.7 |               65.9 |                57.4 |                       2   |                       4.1 |
| AF-Whisper         |               63.7 |               68.3 |                45.2 |                       3.7 |                       7.2 |
| w/o AudioSkills-XL |               56.1 |               42.1 |                14.3 |                       1.6 |                       3.6 |
| Audio Flamingo 3   |               75.8 |               74.4 |                66.9 |                       1.5 |                       3.1 |

## 7 Conclusion, Limitations and Future Work

In this paper, we introduce Audio Flamingo 3, the most capable and open LALM. Our model leverages a custom Whisper, novel data curation techniques, and a 5-stage curriculum learning strategy. Audio Flamingo 3 not only achieves SOTA performance in audio understanding and reasoning but also introduces capabilities, including multi-turn multi-audio chat, on-demand thinking, and voice chat. We detail our practices, including architecture, training, inference, and the evaluation pipeline, and open-source two large datasets. For future work, we aim to address current limitations, including: (1) mitigating the need for a cascaded system for voice chat, (2) making AF3 multi-lingual, and (3) reducing dependency on closed-source models for synthetic data.

## References

- [1] A. Abouelenin, A. Ashfaq, A. Atkinson, H. Awadalla, N. Bach, J. Bao, A. Benhaim, M. Cai, V. Chaudhary, C. Chen, et al. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743 , 2025.
- [2] S. Abu-El-Haija, N. Kothari, J. Lee, P. Natsev, G. Toderici, B. Varadarajan, and S. Vijayanarasimhan. Youtube-8m: A large-scale video classification benchmark. arXiv preprint arXiv:1609.08675 , 2016.
- [3] A. Agostinelli, T. I. Denk, Z. Borsos, J. Engel, M. Verzetti, A. Caillon, Q. Huang, A. Jansen, A. Roberts, M. Tagliasacchi, et al. Musiclm: Generating music from text. arXiv preprint arXiv:2301.11325 , 2023.
- [4] P. Anastassiou, J. Chen, J. Chen, Y. Chen, Z. Chen, Z. Chen, J. Cong, L. Deng, C. Ding, L. Gao, et al. Seed-tts: A family of high-quality versatile speech generation models. arXiv preprint arXiv:2406.02430 , 2024.
- [5] R. Ardila, M. Branson, K. Davis, M. Henretty, M. Kohler, J. Meyer, R. Morais, L. Saunders, F. M. Tyers, and G. Weber. Common voice: A massively-multilingual speech corpus. In Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020) , pages 4211-4215, 2020.
- [6] J. Bai, H. Liu, M. Wang, D. Shi, W. Wang, M. D. Plumbley, W.-S. Gan, and J. Chen. Audiosetcaps: Enriched audio captioning dataset generation using large audio language models. In Audio Imagination: NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation , 2024.
- [7] P. Barros, N. Churamani, E. Lakomkin, H. Siqueira, A. Sutherland, and S. Wermter. The omg-emotion behavior dataset. In 2018 International Joint Conference on Neural Networks (IJCNN) , pages 1-7. IEEE, 2018.
- [8] T. Bertin-Mahieux, D. P. Ellis, B. Whitman, and P. Lamere. The million song dataset. In Proceedings of the 12th International Conference on Music Information Retrieval (ISMIR 2011) , 2011.
- [9] T. Bertin-Mahieux, D. P. Ellis, B. Whitman, and P. Lamere. The million song dataset. In Ismir , volume 2, page 10, 2011.
- [10] R. M. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam, and J. P. Bello. Medleydb: A multitrack dataset for annotation-intensive mir research. In Ismir , volume 14, pages 155-160, 2014.
- [11] C. Busso, M. Bulut, C.-C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. N. Chang, S. Lee, and S. S. Narayanan. Iemocap: Interactive emotional dyadic motion capture database. Language resources and evaluation , 42:335-359, 2008.
- [12] M. Cartwright, J. Cramer, A. E. M. Mendez, Y. Wang, H.-H. Wu, V. Lostanlen, M. Fuentes, G. Dove, C. Mydlarz, J. Salamon, et al. Sonyc-ust-v2: An urban sound tagging dataset with spatiotemporal context. arXiv preprint arXiv:2009.05188 , 2020.
- [13] C. Chen, P. Peng, A. Baid, Z. Xue, W.-N. Hsu, D. Harwath, and K. Grauman. Action2sound: Ambient-aware generation of action sounds from egocentric videos. In European Conference on Computer Vision , pages 277-295. Springer, 2024.
- [14] G. Chen, S. Chai, G. Wang, J. Du, W.-Q. Zhang, C. Weng, D. Su, D. Povey, J. Trmal, J. Zhang, et al. Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio. arXiv preprint arXiv:2106.06909 , 2021.
- [15] H. Chen, W. Xie, A. Vedaldi, and A. Zisserman. Vggsound: A large-scale audio-visual dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 721-725. IEEE, 2020.

- [16] S. Chen, Y. Wu, C. Wang, S. Liu, D. Tompkins, Z. Chen, and F. Wei. Beats: Audio pre-training with acoustic tokenizers, 2022.
- [17] Y. Chen, F. Xue, D. Li, Q. Hu, L. Zhu, X. Li, Y. Fang, H. Tang, S. Yang, Z. Liu, et al. Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188 , 2024.
- [18] Y. Chen, X. Yue, C. Zhang, X. Gao, R. T. Tan, and H. Li. Voicebench: Benchmarking llm-based voice assistants. arXiv preprint arXiv:2410.17196 , 2024.
- [19] Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv, J. He, J. Lin, C. Zhou, and J. Zhou. Qwen2-audio technical report, 2024.
- [20] Y. Chu, J. Xu, X. Zhou, Q. Yang, S. Zhang, Z. Yan, C. Zhou, and J. Zhou. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models, 2023.
- [21] J. S. Chung, A. Nagrani, and A. Zisserman. Voxceleb2: Deep speaker recognition. arXiv preprint arXiv:1806.05622 , 2018.
- [22] C. Cieri, D. Miller, and K. Walker. The fisher corpus: A resource for the next generations of speech-to-text. In LREC , volume 4, pages 69-71, 2004.
- [23] A. Clifton, S. Reddy, Y. Yu, A. Pappu, R. Rezapour, H. Bonab, M. Eskevich, G. Jones, J. Karlgren, B. Carterette, and R. Jones. 100,000 podcasts: A spoken English document corpus. In Proceedings of the 28th International Conference on Computational Linguistics , pages 5903-5917, Barcelona, Spain (Online), Dec. 2020. International Committee on Computational Linguistics.
- [24] F. Daniel, M. Matera, V. Zaccaria, and A. Dell'Orto. Toward truly personal chatbots: on the development of custom conversational assistants. In Proceedings of the 1st international workshop on software engineering for cognitive services , pages 31-36, 2018.
- [25] M. Defferrard, K. Benzi, P. Vandergheynst, and X. Bresson. Fma: A dataset for music analysis. arXiv preprint arXiv:1612.01840 , 2016.
- [26] Z. Deng, Y. Ma, Y. Liu, R. Guo, G. Zhang, W. Chen, W. Huang, and E. Benetos. Musilingo: Bridging music and text with pre-trained language models for music captioning and query response. arXiv preprint arXiv:2309.08730 , 2023.
- [27] S. Deshmukh, B. Elizalde, R. Singh, and H. Wang. Pengi: An audio language model for audio tasks, 2023.
- [28] S. Deshmukh, B. Elizalde, and H. Wang. Audio retrieval with wavtext5k and clap training. arXiv preprint arXiv:2209.14275 , 2022.
- [29] S. Deshmukh, S. Han, H. Bukhari, B. Elizalde, H. Gamper, R. Singh, and B. Raj. Audio entailment: Assessing deductive reasoning for audio understanding. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 39, pages 23769-23777, 2025.
- [30] S. Doh, K. Choi, J. Lee, and J. Nam. Lp-musiccaps: Llm-based pseudo music captioning. arXiv preprint arXiv:2307.16372 , 2023.
- [31] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, and H. Jégou. The faiss library. arXiv preprint arXiv:2401.08281 , 2024.
- [32] K. Drossos, S. Lipping, and T. Virtanen. Clotho: An audio captioning dataset. In ICASSP 20202020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 736-740. IEEE, 2020.
- [33] B. Elizalde, S. Deshmukh, M. Al Ismail, and H. Wang. Clap: Learning audio concepts from natural language supervision, 2022.
- [34] J. Engel, C. Resnick, A. Roberts, S. Dieleman, M. Norouzi, D. Eck, and K. Simonyan. Neural audio synthesis of musical notes with wavenet autoencoders. In International conference on machine learning , pages 1068-1077. PMLR, 2017.

- [35] J. Feng, Q. Sun, C. Xu, P. Zhao, Y. Yang, C. Tao, D. Zhao, and Q. Lin. Mmdialog: A large-scale multi-turn dialogue dataset towards multi-modal open-domain conversation. arXiv preprint arXiv:2211.05719 , 2022.
- [36] E. Fonseca, X. Favory, J. Pons, F. Font, and X. Serra. Fsd50k: an open dataset of humanlabeled sound events. IEEE/ACM Transactions on Audio, Speech, and Language Processing , 30:829-852, 2021.
- [37] E. Fonseca, J. Pons, X. Favory, F. Font, D. Bogdanov, A. Ferraro, S. Oramas, A. Porter, and X. Serra. Freesound datasets: A platform for the creation of open audio datasets. In ISMIR , pages 486-493, 2017.
- [38] P. Foster, S. Sigtia, S. Krstulovic, J. Barker, and M. D. Plumbley. Chime-home: A dataset for sound source recognition in a domestic environment. In 2015 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) , pages 1-5. IEEE, 2015.
- [39] J. F. Gemmeke, D. P. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP) , pages 776-780. IEEE, 2017.
- [40] S. Ghosh, Z. Kong, S. Kumar, S. Sakshi, J. Kim, W. Ping, R. Valle, D. Manocha, and B. Catanzaro. Audio flamingo 2: An audio-language model with long-audio understanding and expert reasoning abilities, 2025.
- [41] S. Ghosh, S. Kumar, A. Seth, C. K. R. Evuru, U. Tyagi, Sakshi, O. Nieto, R. Duraiswami, and D. Manocha. Gama: A large audio-language model with advanced audio understanding and complex reasoning abilities, 2024.
- [42] S. Ghosh, A. Seth, S. Kumar, U. Tyagi, C. K. R. Evuru, S. Ramaneswaran, S. Sakshi, O. Nieto, R. Duraiswami, and D. Manocha. Compa: Addressing the gap in compositional reasoning in audio-language models. In The Twelfth International Conference on Learning Representations .
- [43] J. J. Godfrey, E. C. Holliman, and J. McDaniel. Switchboard: Telephone speech corpus for research and development. In Acoustics, speech, and signal processing, ieee international conference on , volume 1, pages 517-520. IEEE Computer Society, 1992.
- [44] A. Goel, Z. Kong, R. Valle, and B. Catanzaro. Audio dialogues: Dialogues dataset for audio and music understanding. arXiv preprint arXiv:2404.07616 , 2024.
- [45] Y. Gong, A. H. Liu, H. Luo, L. Karlinsky, and J. Glass. Joint audio and speech understanding, 2023.
- [46] Y. Gong, H. Luo, A. H. Liu, L. Karlinsky, and J. Glass. Listen, think, and understand, 2023.
- [47] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
- [48] A. Guzhov, F. Raue, J. Hees, and A. Dengel. Audioclip: Extending clip to image, text and audio, 2021.
- [49] F. Hernandez, V. Nguyen, S. Ghannay, N. Tomashenko, and Y. Esteve. Ted-lium 3: Twice as much data and corpus repartition for experiments on speaker adaptation. In Speech and Computer: 20th International Conference, SPECOM 2018, Leipzig, Germany, September 18-22, 2018, Proceedings 20 , pages 198-208. Springer, 2018.
- [50] S. Hershey, D. P. Ellis, E. Fonseca, A. Jansen, C. Liu, R. C. Moore, and M. Plakal. The benefit of temporally-strong labels in audio event classification. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 366370. IEEE, 2021.
- [51] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al. Lora: Low-rank adaptation of large language models. ICLR , 1(2):3, 2022.

- [52] A. Huang, B. Wu, B. Wang, C. Yan, C. Hu, C. Feng, F. Tian, F. Shen, J. Li, M. Chen, et al. Step-audio: Unified understanding and generation in intelligent speech interaction. arXiv preprint arXiv:2502.11946 , 2025.
- [53] R. Huang, M. Li, D. Yang, J. Shi, X. Chang, Z. Ye, Y. Wu, Z. Hong, J. Huang, J. Liu, Y. Ren, Z. Zhao, and S. Watanabe. Audiogpt: Understanding and generating speech, music, sound, and talking head, 2023.
- [54] A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276 , 2024.
- [55] M. M. Islam, N. Ho, X. Yang, T. Nagarajan, L. Torresani, and G. Bertasius. Video recap: Recursive captioning of hour-long videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 18198-18208, 2024.
- [56] J. James, L. Tian, and C. I. Watson. An open source emotional speech corpus for human robot interaction applications. In Interspeech , pages 2768-2772, 2018.
- [57] I.-Y. Jeong and J. Park. Cochlscene: Acquisition of acoustic scene data using crowdsourcing. In 2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) , pages 17-21. IEEE, 2022.
- [58] X. Ju, Y. Gao, Z. Zhang, Z. Yuan, X. Wang, A. Zeng, Y. Xiong, Q. Xu, and Y. Shan. Miradata: A large-scale video dataset with long durations and structured captions. Advances in Neural Information Processing Systems , 37:48955-48970, 2024.
- [59] W. Kang, X. Yang, Z. Yao, F. Kuang, Y. Yang, L. Guo, L. Lin, and D. Povey. Libriheavy: A 50,000 hours asr corpus with punctuation casing and context. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 10991-10995. IEEE, 2024.
- [60] C. D. Kim, B. Kim, H. Lee, and G. Kim. Audiocaps: Generating captions for audios in the wild. In NAACL-HLT , 2019.
- [61] J. Kim, T. Moon, K. Lee, and J. Cho. Efficient generative modeling with residual vector quantization-based tokens. arXiv preprint arXiv:2412.10208 , 2024.
- [62] P. Koehn. Europarl: A parallel corpus for statistical machine translation. In Proceedings of machine translation summit x: papers , pages 79-86, 2005.
- [63] A. S. Koepke, A.-M. Oncescu, J. F. Henriques, Z. Akata, and S. Albanie. Audio retrieval with natural language queries: A benchmark study. IEEE Transactions on Multimedia , 25:26752685, 2022.
- [64] Y. Koizumi, H. Zen, S. Karita, Y. Ding, K. Yatabe, N. Morioka, M. Bacchiani, Y. Zhang, W. Han, and A. Bapna. Libritts-r: A restored multi-speaker text-to-speech corpus. INTERSPEECH 2023 , 2023.
- [65] Z. Kong, A. Goel, R. Badlani, W. Ping, R. Valle, and B. Catanzaro. Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities, 2024.
- [66] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar. High-fidelity audio compression with improved rvqgan. In Thirty-seventh Conference on Neural Information Processing Systems .
- [67] E. Law, K. West, M. Mandel, M. Bay, and J. Downie. Evaluation of algorithms using games: the case of music annotation. In Proceedings of the 11th International Society for Music Information Retrieval Conference (ISMIR). Utrecht, the Netherlands , 2010.
- [68] C. Lee, R. Roy, M. Xu, J. Raiman, M. Shoeybi, B. Catanzaro, and W. Ping. Nv-embed: Improved techniques for training llms as generalist embedding models. arXiv preprint arXiv:2405.17428 , 2024.

- [69] D. Lee, C. Kim, S. Kim, M. Cho, and W.-S. Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 11523-11532, 2022.
- [70] K. Lee, D. W. Kim, J. Kim, S. Chung, and J. Cho. DiTTo-TTS: Diffusion transformers for scalable text-to-speech without domain-specific factors. In The Thirteenth International Conference on Learning Representations , 2025.
- [71] K. Lee, K. Park, and D. Kim. Dailytalk: Spoken dialogue dataset for conversational textto-speech. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1-5. IEEE, 2023.
- [72] S. Leng, Y. Xing, Z. Cheng, Y. Zhou, H. Zhang, X. Li, D. Zhao, S. Lu, C. Miao, and L. Bing. The curse of multi-modalities: Evaluating hallucinations of large multimodal models across language, visual, and audio. arXiv preprint arXiv:2410.12787 , 2024.
- [73] G. Li, J. Liu, H. Dinkel, Y. Niu, J. Zhang, and J. Luan. Reinforcement learning outperforms supervised fine-tuning: A case study on audio question answering. arXiv preprint arXiv:2503.11197 , 2025.
- [74] G. Li, Y. Wei, Y. Tian, C. Xu, J.-R. Wen, and D. Hu. Learning to answer questions in dynamic audio-visual scenarios. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 19108-19118, 2022.
- [75] T. Li, J. Liu, T. Zhang, Y. Fang, D. Pan, M. Wang, Z. Liang, Z. Li, M. Lin, G. Dong, et al. Baichuan-audio: A unified framework for end-to-end speech interaction. arXiv preprint arXiv:2502.17239 , 2025.
- [76] S. Lipping, P. Sudarsanam, K. Drossos, and T. Virtanen. Clotho-aqa: A crowdsourced dataset for audio question answering. In 2022 30th European Signal Processing Conference (EUSIPCO) , pages 1140-1144. IEEE, 2022.
- [77] S. Liu, H. J. Cho, M. Freedman, X. Ma, and J. May. Recap: retrieval-enhanced context-aware prefix encoder for personalized dialogue response generation. arXiv preprint arXiv:2306.07206 , 2023.
- [78] S. Liu, A. S. Hussain, C. Sun, and Y. Shan. Music understanding llama: Advancing textto-music generation with question answering and captioning. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 286290. IEEE, 2024.
- [79] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 11976-11986, 2022.
- [80] Z. Ma, Z. Chen, Y. Wang, E. S. Chng, and X. Chen. Audio-cot: Exploring chain-of-thought reasoning in large audio language model. arXiv preprint arXiv:2501.07246 , 2025.
- [81] Z. Ma, Y. Ma, Y. Zhu, C. Yang, Y.-W. Chao, R. Xu, W. Chen, Y. Chen, Z. Chen, J. Cong, K. Li, K. Li, S. Li, X. Li, X. Li, Z. Lian, Y . Liang, M. Liu, Z. Niu, T. Wang, Y . Wang, Y . Wang, Y. Wu, G. Yang, J. Yu, R. Yuan, Z. Zheng, Z. Zhou, H. Zhu, W. Xue, E. Benetos, K. Yu, E.-S. Chng, and X. Chen. Mmar: A challenging benchmark for deep reasoning in speech, audio, music, and their mix, 2025.
- [82] L. Martinez-Lucas, M. Abdelwahab, and C. Busso. The msp-conversation corpus. Interspeech 2020 , 2020.
- [83] X. Mei, C. Meng, H. Liu, Q. Kong, T. Ko, C. Zhao, M. D. Plumbley, Y. Zou, and W. Wang. Wavcaps: A chatgpt-assisted weakly-labelled audio captioning dataset for audio-language multimodal research. IEEE/ACM Transactions on Audio, Speech, and Language Processing , 2024.
- [84] J. Melechovsky, Z. Guo, D. Ghosal, N. Majumder, D. Herremans, and S. Poria. Mustango: Toward controllable text-to-music generation. arXiv preprint arXiv:2311.08355 , 2023.

- [85] A. Mesaros, T. Heittola, and T. Virtanen. A multi-device dataset for urban acoustic scene classification. arXiv preprint arXiv:1807.09840 , 2018.
- [86] J. Moon, Y. Kong, and K. H. Chon. Language-independent sleepy speech detection. In 2022 44th Annual International Conference of the IEEE Engineering in Medicine &amp; Biology Society (EMBC) , pages 1981-1984. IEEE, 2022.
- [87] I. M. Morato and A. Mesaros. Diversity and bias in audio captioning datasets. In Detection and Classication of Acoustic Scenes and Events , pages 90-94, 2021.
- [88] M. R. Morris, J. Sohl-Dickstein, N. Fiedel, T. Warkentin, A. Dafoe, A. Faust, C. Farabet, and S. Legg. Position: Levels of agi for operationalizing progress on the path to agi. In Forty-first International Conference on Machine Learning , 2024.
- [89] A.-M. Oncescu, A. Koepke, J. F. Henriques, Z. Akata, and S. Albanie. Audio retrieval with natural language queries. arXiv preprint arXiv:2105.02192 , 2021.
- [90] P. K. O'Neill, V. Lavrukhin, S. Majumdar, V. Noroozi, Y. Zhang, O. Kuchaiev, J. Balam, Y. Dovzhenko, K. Freyberg, M. D. Shulman, et al. Spgispeech: 5,000 hours of transcribed financial audio for fully formatted end-to-end speech recognition. arXiv preprint arXiv:2104.02014 , 2021.
- [91] Z. Ouyang, J.-C. Wang, D. Zhang, B. Chen, S. Li, and Q. Lin. Mqad: A large-scale question answering dataset for training music large language models. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1-5. IEEE, 2025.
- [92] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP) , pages 5206-5210. IEEE, 2015.
- [93] W. Peebles and S. Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision , pages 4195-4205, 2023.
- [94] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, and R. Mihalcea. Meld: A multimodal multi-party dataset for emotion recognition in conversations. arXiv preprint arXiv:1810.02508 , 2018.
- [95] V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert. Mls: A large-scale multilingual dataset for speech research. arXiv preprint arXiv:2012.03411 , 2020.
- [96] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever. Robust speech recognition via large-scale weak supervision, 2022.
- [97] Z. Rafii, A. Liutkus, F.-R. Stöter, S. I. Mimilakis, and R. Bittner. The musdb18 corpus for music separation. 2017.
- [98] M. A. Rahman, Z. I. A. Hakim, N. H. Sarker, B. Paul, and S. A. Fattah. Sonics: Synthetic or not-identifying counterfeit songs. arXiv preprint arXiv:2408.14080 , 2024.
- [99] M. M. Rashid, G. Li, and C. Du. Nonspeech7k dataset: Classification and analysis of human non-speech sound. IET Signal Processing , 17(6):e12233, 2023.
- [100] A. Rousseau, P. Deléglise, and Y. Esteve. Ted-lium: an automatic speech recognition dedicated corpus. In LREC , pages 125-129, 2012.
- [101] S. Sakshi, U. Tyagi, S. Kumar, A. Seth, R. Selvakumar, O. Nieto, R. Duraiswami, S. Ghosh, and D. Manocha. Mmau: A massive multi-task audio understanding and reasoning benchmark. arXiv preprint arXiv:2410.19168 , 2024.
- [102] I. A. P. Santana, F. Pinhelli, J. Donini, L. Catharin, R. B. Mangolin, V. D. Feltrim, M. A. Domingues, et al. Music4all: A new music database and its applications. In 2020 International Conference on Systems, Signals and Image Processing (IWSSIP) , pages 399-404. IEEE, 2020.

- [103] H. Siuzdak. Vocos: Closing the gap between time-domain and fourier-based neural vocoders for high-quality audio synthesis. In The Twelfth International Conference on Learning Representations .
- [104] C. Tang, W. Yu, G. Sun, X. Chen, T. Tan, W. Li, L. Lu, Z. Ma, and C. Zhang. Salmonn: Towards generic hearing abilities for large language models, 2023.
- [105] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 , 2023.
- [106] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 , 2023.
- [107] C. Wang, M. Riviere, A. Lee, A. Wu, C. Talnikar, D. Haziza, M. Williamson, J. Pino, and E. Dupoux. Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. arXiv preprint arXiv:2101.00390 , 2021.
- [108] D. Wang, J. Wu, J. Li, D. Yang, X. Chen, T. Zhang, and H. Meng. Mmsu: A massive multi-task spoken language understanding and reasoning benchmark. arXiv preprint arXiv:2506.04779 , 2025.
- [109] Y. Wang, S. Wu, Y. Zhang, S. Yan, Z. Liu, J. Luo, and H. Fei. Multimodal chain-of-thought reasoning: A comprehensive survey. arXiv preprint arXiv:2503.12605 , 2025.
- [110] B. Weck, I. Manco, E. Benetos, E. Quinton, G. Fazekas, and D. Bogdanov. Muchomusic: Evaluating music understanding in multimodal audio-language models. arXiv preprint arXiv:2408.01337 , 2024.
- [111] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-ofthought prompting elicits reasoning in large language models. Advances in neural information processing systems , 35:24824-24837, 2022.
- [112] H.-H. Wu, P. Seetharaman, K. Kumar, and J. P. Bello. Wav2clip: Learning robust audio representations from clip, 2021.
- [113] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1-5. IEEE, 2023.
- [114] Z. Xie, M. Lin, Z. Liu, P. Wu, S. Yan, and C. Miao. Audio-reasoner: Improving reasoning capability in large audio language models. arXiv preprint arXiv:2503.02318 , 2025.
- [115] Z. Xie and C. Wu. Mini-omni: Language models can hear, talk while thinking in streaming. arXiv preprint arXiv:2408.16725 , 2024.
- [116] G. Xu, P. Jin, L. Hao, Y. Song, L. Sun, and L. Yuan. Llava-o1: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440 , 2024.
- [117] J. Xu, Z. Guo, J. He, H. Hu, T. He, S. Bai, K. Chen, J. Wang, Y. Fan, K. Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215 , 2025.
- [118] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115 , 2024.
- [119] Y. Yuan, D. Jia, X. Zhuang, Y. Chen, Z. Chen, Y. Wang, Y. Wang, X. Liu, X. Kang, M. D. Plumbley, et al. Sound-vecaps: Improving audio generation with visually enhanced captions. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 1-5. IEEE, 2025.
- [120] Y. Zang, S. O'Brien, T. Berg-Kirkpatrick, J. McAuley, and Z. Novack. Are you really listening? boosting perceptual awareness in music-qa benchmarks. arXiv preprint arXiv:2504.00369 , 2025.

- [121] Z. Zhao, Y. Jiang, H. Liu, Y. Wang, and Y. Wang. Librisqa: Advancing free-form and open-ended spoken question answering with a novel dataset and framework. arXiv preprint arXiv:2308.10390 , 2023.
- [122] C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y . Mao, X. Ma, A. Efrat, P. Yu, L. Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems , 36:55006-55021, 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Detailed experiments and evaluation in Section 6 support our claims.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations of our approach are discussed in Section 7.

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

Answer: [NA]

Justification: There is no theoretical result in the paper.

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

Justification: We give details about datasets, experimental setup and training hyperparameters in Appendix H.

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

## Answer: [Yes]

Justification: All the code, data and models are open sourced at https://github.com/ NVIDIA/audio-flamingo/tree/audio\_flamingo\_3 .

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

Justification: We provide all the settings in Appendix H.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: It is expensive to run multiple experiments and report such metrics.

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

Justification: We provide all the details in Appendix H.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Yes, the paper conforms to the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the impact of our work in Appendix K.

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

Justification: Our paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We provide all the details in Appendix G.

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

Answer: [Yes]

Justification: Details about all datasets, including the method for curation, are provided in the main paper. More detailed statistics and examples are provided in the Appendix.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [Yes]

Justification: We provide details about this in Appendix E.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [Yes]

Justification: Most of the human manual analysis was done by the authors of the paper, except chat evaluation, for which we obtained IRB approval from our institution. More details will be provided in the camera-ready.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: LLMs have been used only to help in writing and writing code for parts of experimentation.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix

## A AF-Whisper

## A.1 Training Details

We train AF-Whisper on 512 NVIDIA A100 80GB GPUs. During training, we use an effective batch size of 1024, the AdamW optimizer (learning rate = 10 -4 , weight decay = 0.1), and train using fp16 precision. We train for 5 epochs on the complete dataset and sample instances randomly from the entire pool for each batch.

## A.2 Training Datasets

Table 5 lists the datasets used to train AF-Whisper. For each dataset, we follow the same process outlined in Section 3 of the main paper: generating transcripts, spoken language characteristics, and audio captions. When available, we incorporate gold-standard metadata for these elements (for e.g., transcripts for LibriSpeech or captions for AudioCaps). GPT-4.1 is prompted to produce the final caption using a format similar to Fig. 50, with a modified exemplar. For extracting spoken language characteristics using AF2, we use the following prompt: 'There is a human speaking in the audio. Describe in detail the characteristics of the spoken utterance, including pitch, emotion, mood, speed, and other speech dynamics.'

Table 5: Statistics of audio-caption datasets used for AF-Whisper training.

| Dataset                                                            | #Audio-Text Pairs             |
|--------------------------------------------------------------------|-------------------------------|
| GigaSpeech (L) [14] Speech-in-Sound Captions ∗ [2] SPGISpeech [90] | 2,266,371 1,999,959 1,966,109 |
| Sound-VECaps [119]                                                 | 1,657,029                     |
| Million Songs Dataset [9]                                          | 1,169,997                     |
| Common Voice 15 [5]                                                | 1,109,689                     |
| MiraData [58]                                                      | 748,320                       |
| Action2sound ∗ [13]                                                | 306,602                       |
| NSynth [34]                                                        | 289,205                       |
| LibriSpeech [92]                                                   | 281,241                       |
| Freesound [37]                                                     | 256,695                       |
| AudioSet Strong ∗ [50]                                             | 216,622                       |
| VGGSound [15]                                                      | 185,161                       |
| VoxPopuli (en) [107]                                               | 177,019                       |
| FMA [25]                                                           | 106,412                       |
| Video Recap [55]                                                   | 64,627                        |
| CochlScene [57]                                                    | 60,855                        |
| Music4All [102]                                                    | 109269                        |
| Switchboard [43]                                                   | 76,652                        |
| FSD50k [36]                                                        | 40,966                        |
| MACS [87]                                                          | 31,675                        |
| BBC 2                                                              | 31,201                        |
| MagnaTagATune [67]                                                 | 25,863                        |
| SoundDescs [63]                                                    | 23,085                        |
| Clotho [32]                                                        | 19,195                        |
| TAU-Urban [85]                                                     | 14,400                        |
| MusicCaps [3]                                                      | 5,479                         |
| WavText5K [28]                                                     | 4,347                         |
| SONICS [98]                                                        | 1,602                         |
| SoundBible 3                                                       | 935                           |
| MUSDB18 [97]                                                       | 276                           |
| Medleydb-Pitch [10]                                                | 103                           |
| Total                                                              |                               |
|                                                                    | 13,246,961                    |

## B AudioSkills-XL

Table 6 provides all details, including statistics and references to prompts we used for generating AudioSkills-XL.

Table 6: Detailed statistics of AudioSkills-XL, categorized into individual reasoning types, together with details on open-source datasets, additional meta-data, and prompts used for QA generation. * indicates that these types are further categorized into skills, and we elaborate on this in Section B.1. Rows not grayed out are the contributions of this paper. Speech QA types are the same as LongAudio-XL and explained in Section 4.2, with examples in Figure 3 and more examples in Appendix C.

| Question Type                                                                                                                                                                                                                                                                                         | Size                                                                         | Datasets Used                                                                                                                                                                                                                                                                       | Meta-Data Used                                                                                                                                                                                                                                                              | Prompt Reference                                                                                                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Temporal + ours Attribute Identification Counting Contextual Sound Event Reasoning Contextual Speech Event Reasoning Information Extraction General Reasoning + ours (only sound) Sound Reasoning* (ours) Music Knowledge* (ours) Music Reasoning* (ours) Speech-in-Sound QA (ours) Speech QA* (ours) | 188K 350K 201K 50K 982K 1,272K 858K 704K 300K 300K 1,000K 1,000K 1,739K 200K | Table 14 in [40] Synthetic Data Table 14 in [40] Table 14 in [40] Table 14 in [40] Table 14 in [40] Table 14 in [40] Table 14 in [40] YouTube8M YouTube8M MusicBench, Music4All, MSD MusicBench, Music4All, MSD Speech-in-Sound Caps (YouTube8M) LibriSpeech, GigaSpeech, VoxCeleb2 | Table 14 in [40] - Table 14 in [40] Table 14 in [40] Table 14 in [40] Table 14 in [40] Table 14 in [40] Table 14 in [40] caption caption captions, dataset-specific meta-data captions, dataset-specific meta-data Caption, Transcripts, Speech Characteristics Transcripts | Table 14 in [40] pythonic Table 14 in [40] Table 14 in [40] Table 14 in [40] Table 14 in [40] Table 14 in [40] Table 14 in [40] Fig. 34 Fig. 49, 48, 47 Fig. 31, 28 Fig. 29, 32, 33 Fig. 26 50 Fig. 19, 18, 16, 25 |

## B.1 Skill-Wise Breakdown

## B.1.1 Music Reasoning

Genre and Style: Focuses on the model's ability to infer musical genre or stylistic influences by analyzing instrumentation, arrangement, and production characteristics.

Mood and Expression: Focuses on how well the model interprets the emotional tone or affective content conveyed by the music, such as melancholy, uplifting, or aggressive moods.

Temporal Relations Between Elements: Focuses on the model's understanding of structural evolution within the music over time, including transitions in energy, tempo, or instrumentation across different sections.

Functional Context: Focuses on the model to link the music with real-world settings or usage contexts (e.g., movie scenes, events), requiring understanding of appropriateness and intent.

Lyrics: Focuses on interpretation of lyrical themes and content where applicable, often demanding a blend of semantic understanding and musical context awareness.

Historical and Cultural Context: Focuses on whether the model can connect musical elements to their broader cultural or historical origins (e.g., jazz fusion, protest music), relying on external world knowledge.

Music Texture: Focuses on knowledge of the audio's timbral and sonic character by evaluating aspects such as the layering of instruments, vocal texture, and overall audio quality. This skill captures how dense, sparse, smooth, or gritty a piece sounds, requiring models to interpret descriptive attributes and production characteristics.

Melody: Focuses on understanding the primary musical contour or thematic tune in the audio. Melody-based QAs evaluate recognition of pitch movement, vocal/instrumental phrasing, and stylistic traits such as ornamentation or melodic structure, encouraging indirect inference over simple labeling.

Rhythm and Tempo: Focuses on the temporal structure of the music, including pulse, beat, speed, and time signature. These questions test whether the model can identify rhythmic complexity, tempo changes, and groove characteristics that define a track's pacing or drive.

Harmony and Chords: Focuses on the models' ability to reason about harmonic progressions and chordal structures that shape the emotional and tonal qualities of the audio. This includes interpreting transitions, key relationships, and compositional patterns in harmony using indirect reasoning from musical cues.

Figure 4: Examples of Music Reasoning and Knowledge Questions from AudioSkills-XL. Additionally, we also illustrate examples of music captions generated for audios in Music4All by prompting GPT-4.1 with metadata obtained from the dataset.

<!-- image -->

General Complex Reasoning QA: Evaluates the model's ability to perform multi-dimensional inference on short music segments by combining musical knowledge, perceptual cues, and contextual understanding. These questions are grounded in rich musical attributes, such as dynamics, structure, genre fusion, narrative cues, emotional evolution, and historical style, and require the model to synthesize diverse information to arrive at the correct answer. This category tests higher-order music comprehension across expressive, structural, technical, and cultural dimensions, aiming to emulate how humans make sense of music beyond surface-level tagging.

## B.1.2 Music Knowledge

Instrumentation: Focuses on the model's ability to recognize the instruments used in the music and how their timbre, arrangement, or presence contributes to the overall sound and suitability for various contexts.

Performance: Focuses on understanding of the vocal or instrumental delivery, including vocal tone, articulation, expression, or the presence of unique performance techniques.

Sound Texture: Focuses on the density and layering of sound, such as sparse vs. rich textures, acoustic vs. electronic timbres, and how these contribute to the sonic identity of the piece.

Metre and Rhythm: Focuses on the temporal structure of the piece, including rhythmic patterns, tempo consistency or variation, and the use of syncopation or groove, which are essential for identifying genre or compositional style.

Melody: Focuses on how the model interprets the musical contour and phrasing of the primary tune, including vocal stylings, tonal range, and melodic progression.

Dynamics and Expression: Focuses on the model's sensitivity to dynamic shifts (e.g., soft to loud passages), expressive techniques, and emotional delivery throughout the performance.

Harmony: Focuses on the model's ability to recognize chord progressions, harmonic structure, and tonal relationships, which contribute to the music's emotional or stylistic impact.

## B.1.3 Sound Reasoning

Speech-in-Sound QA: Focuses on reasoning over spoken content in addition to ambient sounds or music to answer complex questions about the input audio, including scene interpretation, action reasoning, etc.

Eco-Acoustic Sounds QA: Focuses on the model's ability to interpret natural environmental conditions based on ambient audio cues. This includes reasoning over weather phenomena such as thunderstorms, snowfall, or rain using non-speech acoustic indicators like wind, water, or animal sounds.

Acoustic Scene Reasoning: Evaluates the model's capability to infer real-world environments from ambient and structural sound patterns. These include background music, reverberation, crowd noise, and electronic elements, enabling scene classification (e.g., arcade, mall, theater) from complex audio mixes.

Sound-Based Event Reasoning: Focuses on identifying and reasoning over specific audio features or events, such as musical motifs, instrument timbres, or recurring sonic patterns, to infer event types or characteristic actions.

Figure 5: Examples of Sound Reasoning QA , together with the metadata used for generating them.

<!-- image -->

Figure 6: Examples of Speech-in-Sound Caps and QA , together with the metadata used for generating them.

<!-- image -->

Figure 7: Examples of general audio QA generated as part of AudioSkills. We generate this as we find models struggle to say a 'No' while responding to questions.

<!-- image -->

## C LongAudio-XL

Tables 9 and 10 present detailed skill-wise statistics for LongAudio-XL, including the source datasets and the minimum, maximum, and average durations of the audio samples.

Below, we also show some examples form LongAudio-Xl in Fig 8

Figure 8: Examples of LongAudio-XL .

<!-- image -->

## D AF-Think

Table 7 provides all details, including statistics and prompts for generating AF-Think.

Table 7: Detailed statistics of AF-Think. Most speech QA examples in this benchmark involve reasoning about ambient sounds in addition to spoken content. As our analysis shows, this added requirement increases task complexity, necessitating deeper inference to answer questions accurately.

| Modality Type   | Size   | Datasets Used                           | Meta-Data Used                            | Prompt Reference   |
|-----------------|--------|-----------------------------------------|-------------------------------------------|--------------------|
| Speech          | 100K   | Speech-in-Sound QA, LongAudio-XL        | transcripts, generated QAs                | Fig. 38, 42        |
| Sound only      | 50K    | AudioSkills-XL (AudioSet-SL, Youtube8M) | captions, QAs, dataset-specific meta-data | Fig. 41, 37        |
| Music           | 100K   | AudioSkills-XL (Music4All, MSD)         | captions, QAs, dataset-specific meta-data | Fig. 39, 40        |

Below, we also provide several examples from AF-Think in Fig. 9:

Figure 9: Examples of AF-Think , for music, speech and sounds.

<!-- image -->

## E AF-Chat

Table 8 provides all details, including statistics and prompts for generating AF-Chat.

Table 8: Detailed statistics of AF-Chat.

| Modality Type       | Size    | Datasets Used            | Meta-Data Used                                                                     | Prompt Reference   |
|---------------------|---------|--------------------------|------------------------------------------------------------------------------------|--------------------|
| Sound &Speech Music | 35K 40K | YouTube8M Music4All, MSD | captions, transcripts, speech characteristics captions, dataset-specific meta-data | Fig. 35 Fig. 36    |

Below, in Fig. 10 and 11 we also provide 2 examples from AF-Chat, one for each sound and music.

## E.1 Human Study for AF-Chat-test

The human verification process has been approved by our institution's Institutional Review Board (IRB). For the human study, we hire 4 Ph.D. students proficient in audio research, including music. For each instance in each test-set dialogue, the students were asked to rate the output of the model on a scale of 1-5 across Factuality (how correct the response is), Usefulness (how useful the response is with respect to the context of the conversation), and Depth (how detailed the response is). For reference, we, the authors of the paper, provide responses scored 1-5 across the 3 aspects. The final score provided in Table 3 is an average of scores across all instances.

## E.2 Clustering for constructing AF-Chat

To construct high-quality multi-turn, multi-audio dialogues for AF-Chat, we implement a targeted clustering strategy that ensures each dialogue is grounded in a semantically diverse but coherent audio context. Rather than sampling audio clips at random, which often leads to incoherent or loosely connected conversations, we curate each dialogue from a controlled pool of semantically related audio samples.

Specifically, for each seed audio, we retrieve its top 8 most semantically similar and top 8 most dissimilar clips from the dataset. Similarity is computed using captions, NV-Embed-v2 embeddings of the captions, and FAISS-based similarity search [31] of the embeddings.

For speech and environmental sounds, we use clips from Speech-in-Sound Caps. For music, we source from Music4All and the Million Song Dataset (MSD). Once the 16-candidate pool is formed (8 similar + 8 dissimilar), we restrict the dialogue construction process to this subset. GPT-4.1 is then

Figure 10: Example of AF-Chat for sound and speech.

<!-- image -->

Figure 11: Example of AF-Chat for music.

<!-- image -->

prompted to construct multi-turn conversations (up to 10 turns) using any combination of these audio clips. This ensures:

1. Topical consistency across turns using similar clips.
2. Diversity and contrast through the inclusion of dissimilar audio.
3. Clear referential structure, as questions may depend on or refer back to earlier audio.

Our clustering strategy was informed by a preliminary human study (participant details similar to Section E.1), where participants engaged in multi-audio, multi-turn conversations with an LALM, focused on tasks such as sound design and music information retrieval. We observed that participants naturally gravitated toward using either highly similar or strongly contrasting audio clips within a dialogue. This behavioral insight motivated our use of similar and dissimilar audio clustering.

Empirically, this approach produced dialogues that were more natural, coherent, and diverse compared to those built from randomly selected audio pools. Moreover, AF3-Chat, when trained on this clustered dataset, outperformed the variant trained on randomly selected audio clips, both in terms of response relevance and conversational depth.

Figure 12: Examples of audio clusters obtained after clustering (Section E.1), used for constructing AF-Chat.

<!-- image -->

## F Prompts

We provide all prompting templates used across our datasets and QA types in Figures 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, and 50.

## G AF3 Training Datasets

Table 11 summarizes all datasets used to train AF3, including total hours, number of audio-QA pairs, and the number of epochs (passes over the dataset) used at each training stage. Similar to [40], we convert all foundational datasets (captioning, classification, etc.) into QA formats, using the same set of prompts for each task mentioned in [40].

## H AF3 Training Details

In this section, we present the training settings of our models across all 5 stages, each with specific configurations. Details are in Table 12.

## I Streaming TTS System Architecture and Training Details

To enable voice output capabilities within our system, we incorporate a text-to-speech (TTS) module that operates on subword text tokens. For efficient and simplified streaming speech synthesis, our TTS module employs a decoder-only architecture.

As illustrated in Fig. 13, the TTS module predicts the subsequent audio token conditioned on incoming subword text tokens from the main AF3 model and the history of previously generated audio tokens. These audio tokens are then decoded into voice output by the neural audio codec. This design simplifies the speech generation pipeline and minimizes latency, which are critical for real-time speech streaming.

## I.1 Neural Audio Codec

We utilize a fully causal convolutional neural audio codec for efficient streaming audio decoding, following [61, 103].

Encoder. Input audio is first resampled to 44.1 kHz. It is then converted into Short-Time Fourier Transform (STFT) parameters using a hop size of 8 and a window size of 32. This STFT representation is processed by an initial 1x1 convolutional layer to produce 384-dimensional hidden embeddings. Following this, the signal undergoes three downsampling stages. Each stage consists of three causal

<!-- image -->

(a) Streaming TTS system architecture.

(b) Iterative unmasking of RVQ audio tokens.

Figure 13: Streaming TTS is enabled by autoregressive audio token generation coupled with a neural audio codec decoder. (a) The streaming TTS system predicts audio tokens conditioned on incoming subword text tokens (e.g., from the main AF3 model) and the history of previously generated audio tokens; these audio tokens are then decoded into voice output by the neural audio codec. (b) The iterative audio token unmasking process relies on an MLP block. This block takes partially masked RVQ tokens and transformer decoder output as input, predicts a cumulative embedding vector, which is subsequently quantized into progressively more unmasked RVQ tokens.

1D-ConvNeXt blocks [79, 103] followed by a strided convolutional layer for downsampling. These strided convolutional layers use a stride and kernel size of 8. Each such layer doubles the hidden dimension, except for the final one, which produces a 512-dimensional output. The encoded output sequence is 4096 times shorter than the raw waveform, corresponding to approximately 10.8 frames per second.

Quantization. The encoded output is quantized into audio tokens using Residual Vector Quantization (RVQ) [69, 66]. The number of RVQ levels is set to 72.

Decoder. The decoder mirrors the encoder's architecture symmetrically, employing 1D transposed convolutional layers for upsampling and causal 1D-ConvNeXt blocks. The final convolutional layer reconstructs the STFT parameters, which are then transformed back into a raw audio waveform via an inverse STFT (iSTFT) similar to Vocos [103].

Training. The codec is trained using a combination of adversarial training and a mel-spectrogram reconstruction loss, following methodologies from DAC [66].

## I.2 Text-to-Speech (TTS) Module

Architecture and Operation. The TTS module's transformer decoder processes a sequence formed by concatenating subword text tokens (from the main AF3 model) and previously generated audio tokens. The resulting hidden states from the transformer serve as conditional input to a multi-layer perceptron (MLP) block. This MLP block then iteratively predicts progressively higher levels of the RVQ tokens, a technique inspired by [61]. In practice, we employ 4 iteration steps during inference. A key aspect contributing to the system's simplicity and low latency is that the model is designed to generate an audio token whenever a text token is emitted by the AF3 model, without requiring explicit alignments between text and speech.

Training and Configuration. During training, the transformer decoder utilizes teacher-forcing with ground-truth audio tokens. The MLP block is trained to estimate the parameters of a mixture-ofgaussians distribution where the number of mixtures is 1024. The objective is to maximize the log-likelihood of predicting the cumulative RVQ token embedding, following [61]. The decoder-only

transformer has a configuration similar to DiT-XL [93, 70]. The MLP block consists of 3 layers, totaling 644 M parameters.

## I.3 Training Data and Processing

The models are trained on a comprehensive dataset comprising LibriTTS-R [64], LibriHeavy [59], the English portion of Multilingual LibriSpeech [95], and proprietary datasets, totaling approximately 100,000 hours of audio. To support long-form speech generation, audio segments from the same speaker are randomly concatenated to create training samples with durations ranging from 1 to 120 seconds.

## J Qualitative Examples

<!-- image -->

. .

. .

. .

<!-- image -->

Prompt: Generate a detailed caption for the input audio, describing all notable speech, sound, and musical events comprehensively. In the caption, transcribe all spoken content by all speakers in the audio precisely.

AF3 Response: A tense, cinematic soundscape unfolds over roughly three minutes. It opens with a woman's surprised question-&lt;Are you being serious right now?= -followed by a man's quick reply, &lt;Yeah.= A door creaks, a camera shutter clicks, and a soft whoosh ushers in a female interviewer who states, &lt; You'd let me interview you as Superman. Sure. Ready? Let's do it, Cronkite. Superman. Miss Lane. Recently, you've come under a lot of fire for what some might-= Her sentence is swallowed by a low electronic beep and a swelling orchestral bed that will underscore the rest of the piece. Over the music she continues, &lt;It's a lot. Today the Secretary of Defense said he was going to look into your actions. That's funny. My actions? I stopped a war. Maybe. Not maybe. I did. In effect, you illegally entered a country. This is how you're going to be? I'm not the one being interviewed, Superman. Did you consult with the president? No. You seemingly acting as a representative of the United States, doing good. I would question myself in the same situation and consider the consequences. People are going to die! Superman!= Her last word is drowned by a sudden explosion and a deep rumble that echoes like a sonic boom. A male voice, urgent and breathless, shouts, &lt;Hey, buddy. Eyes up here.= A brief pause, then a softer male voice reflects, &lt;Your choices. Your actions. That's what makes you who you are. Superman. He's not a man. He's an it. He's somehow become the focal point of the entire world's conversation. I will not accept that. I cleaned your boots. I'll go get them for you.= A door creaks open, a woman asks, &lt;You have a dog?= and the door slams shut. The music swells again, a single word-&lt;you= -is spoken, and the score resolves into a final, triumphant chord that fades to silence.

Figure 14: Demonstration of AF3's capabilities on an audio captioning task. We prompt AF3 with an unseen audio clip-extracted from the Superman 2025 trailer ( https://www.youtube.com/watch?v= 2woCZg5QdVE )-captured in the wild. The model accurately identifies and describes background sounds, spoken content, speaker turns, and transcriptions, demonstrating strong audio understanding. Beyond this example, AF3 supports significantly more complex reasoning tasks. We invite readers to explore these capabilities via our public demo: https://huggingface.co/spaces/nvidia/audio-flamingo-3 .

Figure 15: Prompt used for generating Topic Relationship QA for LongAudioXL.

<!-- image -->

Table 9: Detailed skill-wise and dataset-wise statistics of LongAudio-XL.

| QA Type                 | Dataset                                | #Instances       | Min Dur.(s)     | Max Dur.(s)          | Avg. Dur.(s)       |
|-------------------------|----------------------------------------|------------------|-----------------|----------------------|--------------------|
| Order                   | VoxPopuli [107]                        | 16,926           | 1.87            | 294.80               | 89.55              |
|                         | LibriSpeech [92]                       | 2,340            | 16.02           | 147.59               | 82.01              |
|                         | MELD [94]                              | 4,135            | 1.06            | 108.01               | 30.42              |
|                         | IEMOCAP [11]                           | 599              | 82.00           | 542.00               | 272.45             |
|                         | EuroParl [62]                          | 11,885           | 2.59            | 176.14               | 69.34              |
|                         | Fisher [22]                            | 25,962           | 33.34           | 240.00               | 136.84             |
|                         | Switchboard [43]                       | 2,702            | 22.81           | 148.96               | 87.38              |
|                         | MultiDialog [85]                       | 27,927           | 1.31            | 499.33               | 135.10             |
|                         | VoxCeleb2 [21]                         | 12,855           | 8.00            | 1273.60              | 71.12              |
| Emotion Ident.          | IEMOCAP [11]                           | 300              | 82.00           | 542.00               | 272.22             |
|                         | MELD [94]                              | 1,847            | 1.78            | 108.01               | 33.20              |
| Emotion Causal Reason.  | IEMOCAP [11]                           | 300              | 82.00           | 542.00               | 272.22             |
|                         | MELD [94]                              | 1,850            | 1.57            | 108.01               | 33.13              |
| Emotion Flip Reason.    | IEMOCAP [11]                           | 299              | 82.00           | 542.00               | 272.62             |
|                         | MELD [94]                              | 1,807            | 1.64            | 108.01               | 33.57              |
| Topic Relation. Reason. | VoxPopuli [107]                        | 13,651           | 3.58            | 240.44               | 97.14              |
|                         | LibriSpeech [92]                       | 1,165            | 16.02           | 147.59               | 82.11              |
|                         | MELD [94]                              | 1,518            | 1.89            | 108.01               | 34.45              |
|                         | IEMOCAP [11]                           | 188              | 82.00           | 542.00               | 270.99             |
|                         | EuroParl [62]                          | 9,381            | 7.97            | 176.14               | 70.14              |
|                         | Fisher [22]                            | 20,453           | 33.34           | 240.00               | 136.10             |
|                         | Switchboard [43]                       | 998              | 24.58           | 148.96               | 90.05              |
|                         | MultiDialog [85]                       | 14,906           | 5.11            | 499.33               | 135.35             |
|                         | DailyTalk [71]                         | 3,141            | 8.05            | 103.66               | 35.32              |
|                         | VoxCeleb2 [21]                         | 5,414            | 8.51            | 1193.60              | 78.96              |
| Sarcasm Ident.          | IEMOCAP [11]                           | 299              | 82.00           | 542.00               | 271.58             |
|                         | MELD [94]                              | 1,958            | 1.10            | 108.01               | 31.82              |
| Summarization           | VoxPopuli [107] LibriSpeech [92]       | 13,913 1,057     | 2.12 16.02      | 294.80 147.59        | 91.38 83.15        |
|                         | MELD [94]                              | 2,803            | 1.84            | 108.01               | 32.92              |
|                         | IEMOCAP [11]                           | 300              | 82.00           | 542.00               | 272.22             |
|                         | EuroParl [62]                          | 8,905            | 6.62            | 176.14               | 70.03              |
|                         | Fisher [22]                            | 15,500           | 0.33            | 240.00               | 135.60             |
|                         | Switchboard [43]                       | 1,346            | 24.58           | 148.96               | 87.60 135.73       |
|                         | MultiDialog [85]                       | 20,838 7,218     | 1.93 8.05       | 499.33 103.66        | 31.42              |
|                         | DailyTalk [71]                         |                  |                 |                      |                    |
|                         | VoxCeleb2 [21]                         | 5,894            | 7.94            | 1193.60              | 70.87              |
|                         | Spotify Podcasts [23]                  | 103920           |                 | 18206.44             |                    |
| Needle QA (IE)          |                                        |                  | 0.06            |                      | 2002.99            |
|                         | EuroParl [62] Fisher [22]              | 18,426 37,779    | 6.57 18.59      | 176.14 240.00        | 70.10              |
|                         | IEMOCAP [11]                           | 542              | 82.00           | 542.00               | 135.99             |
|                         | LibriSpeech [92]                       | 2,248            | 16.02           | 147.59               | 272.03 82.82       |
|                         | [23]                                   |                  |                 |                      | 2002.99            |
|                         |                                        |                  | 0.06            | 18206.44             |                    |
|                         | Spotify Podcasts                       | 103920           |                 |                      |                    |
| Response QA (IE)        | VoxPopuli [107] MELD [94] IEMOCAP [11] | 13,913 1,660 177 | 2.12 1.57 82.00 | 294.80 108.01 542.00 | 91.38 31.83 272.52 |
|                         | MultiDialog [85]                       |                  |                 |                      | 135.40             |
|                         | DailyTalk [71]                         | 13,505 4,516     | 1.95 5.72       | 499.33 103.66        | 30.91              |
|                         | Switchboard [43]                       | 862              | 22.81           | 148.96               | 88.75              |

Table 10: Detailed skill-wise and dataset-wise statistics of LongAudio-XL.

| QA Type        | Dataset          | #Instances   |   Min Dur. (s) |   Max Dur.(s) |   Avg. Dur.(s) |
|----------------|------------------|--------------|----------------|---------------|----------------|
| Causal QA (IE) | VoxPopuli [107]  | 12,264       |           4.1  |        240.44 |          92.88 |
|                | LibriSpeech [92] | 1,166        |          16.02 |        147.59 |          82.04 |
|                | MELD [94]        | 2,957        |           1.27 |        108.01 |          31.74 |
|                | IEMOCAP [11]     | 298          |          82    |        542    |         273.1  |
|                | EuroParl [62]    | 7,457        |           7.97 |        176.14 |          70.24 |
|                | Fisher [22]      | 19,335       |          37.17 |        240    |         135.87 |
|                | Switchboard [43] | 1,352        |          22.81 |        148.96 |          87.4  |
|                | MultiDialog [85] | 20,811       |           3.17 |        499.33 |         135.62 |
|                | DailyTalk [71]   | 7,368        |           8.05 |        103.66 |          31.15 |
|                | VoxCeleb2 [21]   | 6,171        |           8.06 |       1193.6  |          71.08 |

You are a helpful AI assistant. You need to act as a question-answer generator for long speech. I will only provide you with the transcript of a single-speaker speech or lecture and you need to  generate  complex  reasoning  based  question-answer  pairs  from  the  conversation.  The  speech might be of diverse nature, which might be a story from an audio book, a parliamentary speech, or some other diverse kind. Specifically, the question should require a listener or the model to understand the contents of the speech and answer about a specific detail in it. My final objective is to train an audio agent with these question-answer pairs to endow it with longspeech understanding and QA abilities. Generate the Question-Answer pairs with the following conditions:

1. The question should require reasoning to be answered correctly.

2. Mention everything abstractly.

3. Do not name the speakers even if a name is provided.

4. The question should ask about a particular detail in the conversation.

5. Most importantly, understand the nature of the speech and generate the QA pair accordingly. Below is an example spoken speech and types of questions/answers below:

mr president the commission thanks the rapporteur mrs regner and the parliament for raising the issue  of  jurisdiction  rules  in  employment  matters.  we  share  the  aim  of  ensuring  strong protection for the rights of employees in general and in particular when they are involved in employment disputes. let me recall that the rights of employees have already been specifically addressed and strengthened in the recent brussels i regulation adopted on six december. two thousand  and  twelve  for  instance  the  new  brussels  i  regulation  provides  for  the  right  for employees  to  sue  several  employers  together  and  the  right  for  employees  to  have  access  to courts  in  europe  even  if  the  employer  is  domiciled  outside  europe.  the  commission  will carefully monitor the application of the new rules over the coming years in the process paying close attention to the employment matters raised by parliament. this can then be considered in the context of the review provided for in article seventy nine of the recast regulation. in this  perspective  the  commission  could  in  the  future  consider  looking  into  the  issue  of employment  contracts  which  has  already  been  mentioned.this  could  be  done  by  looking  more specifically into the suggested fallback clause in article twenty one in cases brought by the employee  against  the  employer  defining  as  relevant  the  place  of  business  from  which  the employee receives or received day to day instructions rather than the place where the business which engaged the employee is situated.

Example Question: In the input speech, on what date did the speaker say that the Brussels I Regulation was adopted?

Example Answer: The speaker mentioned that the Brussels I Regulation was adopted on December 6, 2012.

Generate two such questions. If you think a good quality cannot be made from the conversation, do not generate a question and only return "None" for both question and answer. Return a JSON in the following format: {"Question 1": Question, "Answer 1": Answer, "Question 2": Question, "Answer 2": Answer}

Only return the JSON and nothing else.

Here is the input transcript of the conversation:

Figure 16: Prompt used for generating Needle QA (Information Extraction type) for LongAudioXL.

Table 11: List of fine pre-training and fine-tuning datasets together with their training composition.

| Dataset                                  | Hours               | Num. Pairs   | St. 1       | St. 2       | St. 3       | St. 3.5   | St. 4   |
|------------------------------------------|---------------------|--------------|-------------|-------------|-------------|-----------|---------|
| AudioSkills-XL (Uurs)                    | -                   | 9700K        | -           | 2.0         | 2.0         | -         |         |
| LongAudioXL (Ours)                       | -                   | 1000K        | 1 . 0       | 1 . 0       | 1 . 0       | 1 . 0     | -       |
| AF-Think (Ours)                          | -                   | 250K         | 1 . 0       | 1 . 0       | 1 . 0       | 2 . 0     | -       |
| AF-Chat (Ours)                           | -                   | 75K          | -           | -           | -           | -         | 1 . 0   |
| CompA-R [42]                             | 159 hrs             | 350k         | -           | 2.0         | 2.0         | -         | -       |
| MusicBench [84]                          | 115.5 hrs           | 686k         | -           | 1.0         | 1.0         | -         | -       |
| Mu-LLAMA [78]                            | 62.9 hrs            | 70k          | 1.0         | 2.0         | 2.0         | -         | -       |
| Salmonn AQA [104]                        | 800 hrs             | 270k         | -           | 1.0         | 1.0         | -         | -       |
| ClothoAQA [76]                           | 7.4 hrs             | 9.7K         | -           | 8 . 0       | 8 . 0       | -         | -       |
| OpenAQA [46]                             | 693.2 hrs           | 1959.8K      | -           | 1 . 0       | 1 . 0       | -         | -       |
| Clotho-v2 [32]                           | 24.0 hrs            | 19.2K        | 1.0         | 2 . 0       | 2 . 0       | -         | -       |
| MACS [87]                                | 10.9 hrs            | 17.3K        | -           | 1 . 0       | 1 . 0       | -         | -       |
| FSD50k [36]                              | 80.8 hrs            | 41.0K        | 1.0         | 1 . 0       | 1 . 0       | -         | -       |
| CochlScene [57]                          | 169.0 hrs           | 60.9K        | -           | 1 . 0       | 1 . 0       | -         | -       |
| NonSpeech 7k [99]                        | 6.2 hrs             | 6.3K         | -           | 4 . 0       | 4 . 0       | -         | -       |
| Chime-home [38]                          | 5.0 hrs             | 4.5K         | -           | 1 . 0       | 1 . 0       | -         | -       |
| Sonyc-UST [12]                           | 34.9 hrs            | 27.9K        | -           | 1 . 0       | 1 . 0       | -         | -       |
| Emov-DB [86]                             | 7.8 hrs             | 6.8K         | -           | 1 . 0       | 1 . 0       | -         | -       |
| JL-Corpus [56]                           | 1.4 hrs             | 2.4K         | -           | 6 . 0       | 6 . 0       | -         | -       |
| Tess                                     | 1.6 hrs             | 2.8K         | -           | 2 . 0       | 2 . 0       | -         |         |
| OMGEmotion [7]                           | 3.0 hrs             | 1.7K         | -           | 3 . 0       | 3 . 0       | -         | -       |
| MusicAVQA audio-only [74]                | 77.1 hrs            | 5.7K         | -           | 6 . 0       | 6 . 0       | -         | -       |
| MusicQA [91]                             | 62.9 hrs            | 70K          | -           | 1 . 0       | 1 . 0       | -         | -       |
| LP-MusicCaps MSD                         | 5805.7 hrs          | 1331.8K      | 1.0         | 1 . 0       | 1 . 0       | -         | -       |
| [30] LP-MusicCaps MTT [30]               | 126.4 hrs           | 46.9K        | 1.0         | 1 . 0       | 1 . 0       | -         | -       |
| LP-MusicCaps MC [30]                     | 7.4 hrs             | 7.9K         | 1.0         | 2 . 0       | 2 . 0       | -         | -       |
| MusicCaps [3]                            | 7.4 hrs             | 2.6K         | 1.0         | 6 . 0       | 6 . 0       | -         | -       |
| NSynth [34]                              | 321.3 hrs           | 289.2K       | -           | 8 . 0       | 8 . 0       | -         | -       |
| MusDB-HQ [97]                            | 29.1 hrs            | 10.2K        | -           | 2 . 0       | 2 . 0       | -         | -       |
| FMA [25]                                 | 860.7 hrs           | 104.2K       | -           | 1 . 0       | 1 . 0       | -         | -       |
| Laion630k BBCSoundEffects [113]          | 456.9 hrs           | 15.1K        | 1 . 0       | -           | 1 . 0       | -         | -       |
| Laion630k Freesound [113]                | 2494.8 hrs          | 306.5K       | 1 . 0       | -           | 1 . 0       | -         | -       |
| SoundDescs [63]                          | 749.7 hrs           | 23.1K        | 1 . 0       | -           | 1 . 0 1 . 0 | -         | -       |
| WavCaps [83]                             | 3793.3 hrs          | 402.6 K      | 1 . 0       | - -         | 1 . 0       | -         | -       |
| AudioSet [39]                            | 2617.8 hrs 23.8 hrs | 950.8K       | 1.0 1 . 0   | -           | 1 . 0       | - -       | - -     |
| WavText5K [28] MSP-Podcast [82]          | 73.9 hrs            | 4.3K 45.1K   | 1 . 0       | 1 . 0       | 1 . 0       | -         | -       |
| MELD [94]                                | 8.7 hrs             | 32.9K        | 1 . 0       | 1 . 0       | 1 . 0       | -         | -       |
| MusicAVQA audio-visual [74]              | 142.4 hrs           | 17.9K        | 1 . 0       | 6 . 0       | 6 . 0       | -         | -       |
| Music4All Captions (ours)                |                     | 55.6K        |             | -           | 1 . 0       |           | -       |
|                                          | 910.5 hrs           |              | 1 . 0       |             |             | -         |         |
| MSD Captions (ours)                      | 15449.9 hrs         | 55.6K        | 1 . 0       | -           | 1 . 0       | -         | -       |
| Speech-in-Sound Captions (ours)          | 6227.6 hrs          | 1999959      | 1 . 0       | -           | 1 . 0       | -         | -       |
| LibriSpeech [92]                         | 960 hrs             | 281.2K       | 1 . 0       | 1 . 0       | 1 . 0       | -         | -       |
| Switchboard [43]                         | 109.9 hrs           | 76.6K        | 1 . 0       | 1 . 0       | 1 . 0       | -         | -       |
| GigaSpeech (L) [14]                      | 2499.8 hrs          | 2266.3K      | 1 . 0       | 1 . 0       | 1 . 0       | -         | -       |
| Common Voice 15 [5] VoxPopuli (en) [107] | 1752.1 hrs          | 1109.6K 177K | 1 . 0 1 . 0 | 1 . 0 1 . 0 | 1 . 0 1 . 0 | - -       | - -     |
| TEDLIUM (en) [49]                        | 501.8 hrs 472.3 hrs | 68K          | 1 . 0       | 1 . 0       | 1 . 0       | -         | -       |
| SPGISpeech [90]                          | 4999.8 hrs          | 1966.1K      | 1 . 0       | 1 . 0       | 1 . 0       | -         | -       |
| VoiceAssistant400K [115]                 | 684 hrs             | 470K         | -           | -           | -           | -         | 1 . 0   |

Table 12: Training settings across stages.

| Settings                                     | Stage1   | Stage2   | Stage3                | Stage3.5   | Stage4   |
|----------------------------------------------|----------|----------|-----------------------|------------|----------|
| per device batch size                        | 64       | 16       | 4                     | 4          | 2        |
| learning rate                                | 1e-3     | 2e-5     | 2e-5                  | 5e-5       | 5e-5     |
| learning schedule warm up ratio weight decay |          |          | Cosine decay 0.03 0.0 |            |          |
| epoch                                        | 1        | 1        | 1                     | 2          | 2        |
| bf16                                         | ✓        | ✓        | ✓                     | ✓          | ✓        |
| grad accumulate                              | 1        | 2        | 4                     | 4          | 8        |
| DeepSpeed stage GPUs                         |          |          | Zero3 128 × A100      |            |          |

You are a helpful AI assistant. You need to act as a question-answer generator for long speech. I will only provide you with the transcript of a single-speaker speech or lecture and you need to  generate  complex  reasoning  based  question-answer  pairs  from  the  conversation.  The  speech might be of diverse nature, which might be a story from an audio book, a parliamentary speech, or some other diverse kind. Specifically, the question should require a listener or the model to understand the contents of the speech and understand the main topic of the speech. My final objective is to train an audio agent with these question-answer pairs to endow it with longspeech understanding and QA abilities. Generate the Question-Answer pairs with the following conditions:

1. The question should require reasoning to be answered correctly.

2. Mention everything abstractly.

3. Do not name the speakers even if a name is provided.

4. Most importantly, understand the nature of the speech and generate the QA pair accordingly. Below is an example spoken speech and types of questions/answers below:

mr president the commission thanks the rapporteur mrs regner and the parliament for raising the issue  of  jurisdiction  rules  in  employment  matters.  we  share  the  aim  of  ensuring  strong protection for the rights of employees in general and in particular when they are involved in employment disputes. let me recall that the rights of employees have already been specifically addressed and strengthened in the recent brussels i regulation adopted on six december. two thousand  and  twelve  for  instance  the  new  brussels  i  regulation  provides  for  the  right  for employees  to  sue  several  employers  together  and  the  right  for  employees  to  have  access  to courts  in  europe  even  if  the  employer  is  domiciled  outside  europe.  the  commission  will carefully monitor the application of the new rules over the coming years in the process paying close attention to the employment matters raised by parliament. this can then be considered in the context of the review provided for in article seventy nine of the recast regulation. in this  perspective  the  commission  could  in  the  future  consider  looking  into  the  issue  of employment  contracts  which  has  already  been  mentioned.this  could  be  done  by  looking  more specifically into the suggested fallback clause in article twenty one in cases brought by the employee  against  the  employer  defining  as  relevant  the  place  of  business  from  which  the employee receives or received day to day instructions rather than the place where the business which engaged the employee is situated.

Example Question: What is the main topic of the input speech?

Example  Answer:  The  speaker  talks  about  enhancing  employee  rights  in  employment  disputes, focusing on jurisdiction rules under the Brussels I regulation and potential future improvements.

Generate one such question.  If you think a good quality cannot be made from the conversation, do not generate a question and only return "None" for both question and answer. Return a JSON in the following format: {"Question 1": Question, "Answer 1": Answer} Only return the JSON and nothing else.

Here is the input transcript of the conversation:

Figure 17: Prompt used for generating Topic QA (Information Extraction type) for LongAudioXL.

You are a helpful AI assistant. You need to act as a question-answer generator for long speech. I will only provide you with the transcript of a single-speaker speech or lecture and you need to  generate  complex  reasoning  based  question-answer  pairs  from  the  conversation.  The  speech might be of diverse nature, which might be a story from an audio book, a parliamentary speech, or some other diverse kind.  Specifically, the question should require a listener or the model to understand the contents of the speech/lecture/story and answer questions regarding the order of topics being talked about in the speech. My final objective is to train an audio agent with these  question-answer  pairs  to  endow  it  with  long-speech  understanding  and  QA  abilities. Generate the Question-Answer pairs with the following conditions:

1. The question should require reasoning to be answered correctly.
2. Mention everything abstractly.
3. Do not name the speakers even if a name is provided.

Below is an example conversation and types of questions/answers below:

mr president the commission thanks the rapporteur mrs regner and the parliament for raising the issue  of  jurisdiction  rules  in  employment  matters.  we  share  the  aim  of  ensuring  strong protection for the rights of employees in general and in particular when they are involved in employment disputes. let me recall that the rights of employees have already been specifically addressed and strengthened in the recent brussels i regulation adopted on six december. two thousand  and  twelve  for  instance  the  new  brussels  i  regulation  provides  for  the  right  for employees  to  sue  several  employers  together  and  the  right  for  employees  to  have  access  to courts  in  europe  even  if  the  employer  is  domiciled  outside  europe.  the  commission  will carefully monitor the application of the new rules over the coming years in the process paying close attention to the employment matters raised by parliament. this can then be considered in the context of the review provided for in article seventy nine of the recast regulation. in this  perspective  the  commission  could  in  the  future  consider  looking  into  the  issue  of employment  contracts  which  has  already  been  mentioned.this  could  be  done  by  looking  more specifically into the suggested fallback clause in article twenty one in cases brought by the employee  against  the  employer  defining  as  relevant  the  place  of  business  from  which  the employee receives or received day to day instructions rather than the place where the business which engaged the employee is situated.

Question (Order type): What is the correct order of topics discussed in the input speech? Options  (Order  type)[(A)  The  Commission  thanks  the  rapporteur  and  Parliament.,  (B)  The  new Brussels I regulation's provisions for employee rights are introduced.,(C) The possibility of future considerations regarding employment contracts is mentioned., (D).The Commission plans to monitor the application of new rules.]

Answer (Order type):(A)(B)(D)(C) - This question can be also phrased as "Arrange the topics in the input speech according to their sequence of occurrence" and then you provide the options and the answers.

Question (Temporal Referring type):  What topic is introduced early in the speech? Options (Temporal Referring type): [(A)The new Brussels I regulation's provisions for employee rights,(B)  Future  considerations  on  employment  contracts,  (C)  The  importance  of  protecting employee rights, (D) Monitoring of the new regulations] Answer (Temporal Referring type): (C) The importance of protecting employee rights

Question (Temporal Grounding type): When in the speech does the speaker discuss the possibility of reviewing employment contracts?

```
Options (Temporal Grounding type):[ (A) the begin (B) the middle, (C) the end]
```

Answer (Temporal Grounding type): (C) The end

Question (Attribute type): How does the tone of the speech shift over time? Answer  (Attribute  type):[(A)  It  becomes  more  speculative  as  potential  future  actions  are discussed., (B) It remains formal and informative throughout., (C) It shifts to an optimistic tone as the new rights are discussed.]

Answer  (Attribute  type):  (A)  It  becomes  more  speculative  as  potential  future  actions  are discussed.

Return  a  JSON  in  the  following  format:  {"Temporal  Referring  Question":  Question,  "Temporal Referring Options": Options, "Temporal Referring Answer": Answer, "Temporal Grounding Question":  Question,  "Temporal  Grounding  Options":  Options,  "Temporal  Grounding  Answer": Answer, "Order Question": Question, "Order Options": Options, "Order Answer": Answer, "Attribute Question": Question, "Attribute Options": Options, "Attribute Answer": Answer}. Only return the JSON and nothing else. If a question type is not possible, don't output and only output "None". Here is the input information:

Figure 18: Prompt used for generating Order QA for LongAudioXL.

You are a helpful AI assistant. You need to act as a question-answer generator for long speech. I will only provide you with the transcript of a single-speaker speech or lecture and you need to  generate  complex  reasoning  based  question-answer  pairs  from  the  conversation.  The  speech might be of diverse nature, which might be a story from an audio book, a parliamentary speech, or some other diverse kind. Specifically, the question should require a listener or the model to  understand  the  contents  of  the  speech  and  infer  the  actual  reason  behind  a  particular statement or utterance made by the speaker. The question should involve the listener or the model to look further than the surface and literal meaning of the speech, ideas, or sentences to find the correct answer. My final objective is to train an audio agent with these questionanswer  pairs  to  endow  it  with  long-speech  understanding  and  QA  abilities.  Generate  the Question-Answer pairs with the following conditions:

1. The question should require reasoning to be answered correctly.
2. Mention everything abstractly.
3. Do not name the speakers even if a name is provided.
4. Most importantly, understand the nature of the speech and generate the QA pair accordingly. Below is an example spoken speech and types of questions/answers below:

mr president the commission thanks the rapporteur mrs regner and the parliament for raising the issue  of  jurisdiction  rules  in  employment  matters.  we  share  the  aim  of  ensuring  strong protection for the rights of employees in general and in particular when they are involved in employment disputes. let me recall that the rights of employees have already been specifically addressed and strengthened in the recent brussels i regulation adopted on six december. two thousand  and  twelve  for  instance  the  new  brussels  i  regulation  provides  for  the  right  for employees  to  sue  several  employers  together  and  the  right  for  employees  to  have  access  to courts  in  europe  even  if  the  employer  is  domiciled  outside  europe.  the  commission  will carefully monitor the application of the new rules over the coming years in the process paying close attention to the employment matters raised by parliament. this can then be considered in the context of the review provided for in article seventy nine of the recast regulation. in this  perspective  the  commission  could  in  the  future  consider  looking  into  the  issue  of employment  contracts  which  has  already  been  mentioned.this  could  be  done  by  looking  more specifically into the suggested fallback clause in article twenty one in cases brought by the employee  against  the  employer  defining  as  relevant  the  place  of  business  from  which  the employee receives or received day to day instructions rather than the place where the business which engaged the employee is situated.

Example  Question:  According  to  the  input  speech,  what  might  the  Commission  consider  in  the future regarding employment contracts based on the talk?

Example Answer: According to the given input speech, the Commission might consider addressing the issue of employment contracts by examining the fallback clause in Article 21, which could involve defining the relevant place of business for day-to-day instructions, rather than the employer's primary business location.

Generate two such questions. If you think a good quality cannot be made from the conversation, do not generate a question and only return "None" for both question and answer. Return a JSON in the following format: {"Question 1": Question, "Answer 1": Answer, "Question 2": Question, "Answer 2": Answer}

Only return the JSON and nothing else.

Here is the input transcript of the conversation:

Figure 19: Prompt used for generating Causal QA for LongAudioXL.

Figure 20: Prompt used for generating Summarization QA (Summary QA) for LongAudioXL.

<!-- image -->

Figure 21: Prompt used for generating Emotion Flip QA (Emotional State Reasoning type) for LongAudioXL.

<!-- image -->

Figure 22: Prompt used for generating Causal Reasoning (Emotional State Reasoning type) for LongAudioXL.

<!-- image -->

Figure 23: Prompt used for generating Sarcasm Identification QA for LongAudioXL.

<!-- image -->

Figure 24: Prompt used for generating Identification QA (Emotional State Reasoning type) for LongAudioXL.

<!-- image -->

Figure 25: Prompt used for generating ResponseQA (Information Extraction type) for LongAudioXL.

<!-- image -->

Figure 26: Prompt used for generating Speech-in-Sound QA for AudioSkills-XL.

<!-- image -->

Figure 27: Prompt used for generating captions for Million Songs Dataset . Noisy captions for the prompt are generated using AF2.

<!-- image -->

Figure 28: Prompt used for generating Music Knowledge QA from Million Songs Dataset for AudioSkills-XL. Noisy captions for the prompt are generated using AF2.

<!-- image -->

You are a helpful AI assistant. You need to act as a question-answer generator for songs and generate reasoning-based Question-Answer pairs. Reasoning-based QAs refer to types of questions that would ask the model to understand an input song (music and vocals), reason about it, and answer  a  question.  My  final  objective  is  to  train  an  audio  understanding  agent  with  these question-answer pairs to endow it with music understanding and QA abilities. To generate this question-answer pair, I will provide you with several meta-data about the input music. The meta data I will provide you is as follows:

title of the song, release of the song, artist name for the song, year the song was launched, and optionally some tags describing the song (may or may not provide). Additionally, I will provide you with a list of captions generated by a music captioning model. The list has music captions generated by a music captioner for every 30 seconds segments of the music and may be noisy  *Please  use  your  world  knowledge  with  all  the  meta-data  I  provide  to  understand  the music, including name of the song, tempo, etc. If you have world knowledge about the music from the aritst information or other meta-data, priortize it over the caption or verufy integrity.

## Generate using the following rules:

1) Generate QA pairs with crisp question and answer. Do not have too much information in the question.  2)  Generate  MCQ  based  questions.  The  choices  should  be  contrastive  that  may  rise confusion to the model.

3) The questions should have a level of difficulty for the model..

You are supposed to generate QAs using a pre-defined set of skills. I will now provide you with the name of skills and an example QA pair for each skill: Reasoning Skills: genre and style Example Question: Which of the following would be an appropriate setting for this music track? Example Options: (A) A romantic comedy montage (B) It features diverse synthesizers (C) Ending credits of a video game or sci-fi movie (D) A motivational fitness workout Example Answer: (B) It features diverse synthesizers

Reasoning Skills: mood and expression

Example Question: Which of the following would be an appropriate setting for this music track? Example Options: (A) A romantic comedy montage (B) It features diverse synthesizers (C) Ending credits of a video game or sci-fi movie (D) A motivational fitness workout Example Answer: (B) It features diverse synthesizers

Reasoning Skills: temporal relations between elements Example Question: How would you describe the energy level of the track? Example Options: (A) It features a sudden tempo change (B) Gradually evolving from dreamy to driving (C) High energy throughout (D) Instrumental electronic piece Example Answer: (D) Instrumental electronic piece

Reasoning Skills: functional context

Example Question: Which of the following would be an appropriate setting for this music track? Example Options: (A) A romantic comedy montage (B) It features diverse synthesizers (C) Ending credits of a video game or sci-fi movie (D) A motivational fitness workout Example Answer: (A) A romantic comedy montage

Reasoning Skills: lyrics

Example Question: What is the main theme of the song's lyrics?

Example Options: (A) Pop-rock (B) Political commentary (C) Happiness (D) Love

Example Answer: (A) Pop-rock

Reasoning Skills: historical and cultural context

Example Question: What unique combination of musical styles is evident in this song?

Example  Options:  (A)  The  song  has  a  fast  tempo  (B)  Jazz  and  country  music  (C)  The  song  is played by a famous band (D) Classical blues chords with big band style

Example Answer:  (B) Jazz and country music

You may also generate QA pairs with multiple skills required.

It is not necessary to generate all knoweldge and reasoning based questions, generate 1-3 QAs for unique skills (or unique combination of skills) from each input for the skills you feel is the best fit and can generate the best quality of questions. If nothing is return None*. Please generate Output a JSON of the following format: {"Question 1 Type": Reasoning Type of Question 1,  "Question  1":  Question  1,  "Answer  1":  Answer  for  the  question  1,  "Options  1":  List  of options for question 1, minimum 4 and contrastive options from the original, "Question 2 Type": Reasoning  Type  of  Question  2,  "Question  2":  Question  2,  "Options  2":  List  of  options  for question 2, minimum 4 and contrastive options from the original, "Answer 2": Answer for the question 2}.If no question is possible, just return None, but don't return wrong QA.

Output a JSON of the similar format and nothing else

Here is the input information:

Figure 29: Prompt used for generating Music Reasoning QA from Million Songs Dataset for AudioSkills-XL. Noisy captions for the prompt are generated using AF2.

Valence: 0.263

Tempo: 172.208 BPM

Duration (ms): 325096

Genres:Underground Hip-Hop

Language: English

Figure 30: Prompt used for generating captions for Music4All . Noisy captions for the prompt are generated using AF2.

<!-- image -->

Figure 31: Prompt used for generating Music Knowledge QA from Music4All for AudioSkills-XL. Noisy captions for the prompt are generated using AF2.

<!-- image -->

Figure 32: Prompt used for generating General Open-Ended Complex Reasoning QA for Music Reasoning QA from Music4All for AudioSkills-XL.

<!-- image -->

Figure 33: Prompt used for generating Music Reasoning QA from Music4All for AudioSkills-XL. Noisy captions for the prompt are generated using AF2.

<!-- image -->

Figure 34: Prompt used for generating Yes-No QA (part of General Reasoning+ours (only sound) from AudioSet-SL) for AudioSkills-XL.

<!-- image -->

```
You are a helpful AI assistant. You need to generate a chat between an AI agent and a curious human about audios. The audios may have speech, sound and music content. I will provide you with various meta-data for about several audio samples. These include: 1) Caption: A speech-aware audio caption that emphasizes on spoken content, and is framed using speech transcription, sound caption (audiocaps style) and characteristics of the spoken content. 2) Audio Caption: An audiocaps style caption that describes only the sounds. 3) Transcript: A transcripted generated using Whisper. 4) Speech Characteristics: Characteristics of the spoken content in the audio, describing pitch, emotion or whatever is relevant. Use the meta-data to understand about the audios. Specifically, I will provide you with a group of positives (audios similar in theme) and a group of negatives (audios dissimilar in theme). Similarity is calculated using semantics and events. You need to generate to a chat between an AI agent and a human, where the human is trying to analyze the audio samples with questions across the audios. My final objective is to train an audio understanding agent with these chat question-answers to endow it with chat and QA abilities. Here are the rules to generate the chat: 1. Take minimum 2 and maximum 10 turns for the conversation, 2. You may use turns that refer to previous audios (however, please refer abstractly and not directly with audio numbers -- see example next). You may also have turns that only have a follow-up question and no audio input. 3: The questions to the AI model should require reasoning, world knowledge and information extraction. 4.  Remember  that  the  AI  model  I  will  train  with  the  chat  QAs  will  only  receive  the  audios  and  not  details  in  text  -  so  generate  QAs accordingly and do not mention anything referring directly from the metadata. 5. You are not required to use all the samples. Your priority is to make a sensible and plausible chat. Use only samples you think best fits the narrative. 6.**Include  a  good  proportion  of  positives  and  negatives  in  the  chat,  intelerleave  them  and  so  on.  Also  do  not  refer  to  earlier  audios directly. Trying having multiple turns for a single input audio. Here is an example: { "Turn 1": { "Human": { "Audio": "Positive 0", "Text": "This guy keeps yelling 'You're a monster!' - it's intense. What's driving that intensity beyond just repetition?" }, "AI": { "Text": "The intensity comes from the rising pitch, relentless rhythm, and minimal pausing. It feels like a panic loop. The acoustics add to it too - it's close-mic'd and slightly compressed, which gives it a suffocating edge, like he's spiraling emotionally in a confined space." } }, "Turn 2": { "Human": { "Text": "Is this kind of panic more emotional or performative, you think?" }, "AI": { "Text": "It feels genuinely emotional. There's no background music or theatrical pacing - just raw, unfiltered distress. The lack of buildup or resolution makes it sound like a real emotional outburst, not a staged one." } }, "Turn 3": { "Human": { "Audio": "Positive 7", "Text": "This one's also someone yelling 'I'm a monster,' but it has music and engine noises. Why does it feel more cinematic?" }, "AI": { "Text": "Because it is - the backing elements like distorted guitars and vehicle sounds create a stylized soundscape. The pacing accelerates a bit too, mimicking a scene's climax. Compared to the earlier clip, this one feels like it's part of a fictional world, not real life." } }, "Turn 4": { "Human": { "Text": "Would you say the emotion is the same between these two?" }, "AI": { "Text": "They're in the same emotional neighborhood - fear, intensity - but the delivery differs. The first feels like someone truly overwhelmed; this one is more like a character mid-breakdown, shaped by the dramatic environment around them." } }, "Turn 5": { "Human": { "Audio": "Positive 2", "Text": "Here, the speaker just says 'The world is a monster,' but it hits different - slower, more deliberate." }, "AI": { "Text": "Exactly. It moves from panic to reflection. The voice is calm but heavy, with ambient crackling that adds a sense of solitude. Unlike the earlier clips full of shouting, this one feels resigned - like someone making sense of things after the storm." } }, "Turn 6": { "Human": { "Text": "Could something like this work as the emotional landing after those earlier, louder clips?" }, "AI": { "Text":  "Absolutely.  It  would  make  a  great  resolution.  Start  with  chaos,  transition  through  stylized  drama,  and  end  with  quiet realization. It forms a psychological arc - from raw emotion to introspection." } } } If no question is possible, just return None, but don't return wrong QA. Output a JSON of the similar format and nothing else Here is the input information:
```

Figure 35: Prompt used for generating multi-turn, multi-audio chat/dialogues (speech and sounds) for AF-Chat.

```
You are a helpful AI assistant. You need to generate a chat between an AI agent and a curios musician. I will provide you with varios metadata for about 3-6 music samples. Use your world-knowledge and other meta-data to understand about the music/song. You need to generate to a chat between an AI agent and a musician, where the musician is trying to analyze the music samples with questions across the music samples. My final objective is to train an audio understanding agent with these chat question-answers to endow it with chat and QA abilities. Here are the rules to generate the chat: 1. Take minimum 2 and maximum 10 turns for the conversation. 2. You may use turns that refer to previous audios (however, please refer abstractly and not directly with audio numbers -- see example next). You may also have turns that only have a follow-up question and no audio input. 3: The questions to the AI model should require reasoning, world knowledge and information extraction. 4. Remember that the AI model I will train with the chat QAs will only receive the music/song and not details in text - so generate QAs accordingly and do not mention anything in the Question from the Human. 5. You are not required to use all the samples. Your priority is to make a sensible and plausible chat. Use only music samples you think best fits the narrative. Here is an example: Example Input: { "Audio 0": { "name": "./Music4ALL/audios/0009fFIM1eYThaPg.mp3", other metadata.. }, "Audio 1": { "name": "./Music4ALL/audios/0010xmHR6UICBOYT.mp3", other metadata.. }, "Audio 2": { "name": "./Music4ALL/audios/002Jyd0vN4HyCpqL.mp3", other metadata.. }, "Audio 3": { "name": "./Music4ALL/audios/006TYKNjNxWjfKjy.mp3", other metadata.. }, "Audio 4": { "name": "./Music4ALL/audios/007LIJOPQ4Sb98qV.mp3", other metadata.. } } Example Output: { "Turn 1": { "Human": { "Audio": "Audio 0", "Text": "This track feels super polished and upbeat, but I wonder - could it work in a darker, post-punk kind of vibe?" }, "AI": { "Text": "You're picking up on the contrast between the clean dance-pop structure of this track and the moodier, raw textures of postpunk. Rhythmically, it could hold up - it's around 110 BPM and steady. If you revoice the synths with analog gear, introduce chorus-laden guitars, and push the vocal through a vintage delay or tape sim, you could easily channel a darker vibe while keeping the groove." } }, "Turn 2": { "Human": { "Text": "Interesting - would you change the harmony too, or just the instrumentation?" }, "AI": { "Text": "Good follow-up. The harmony could stay largely the same if you're going for contrast, but shifting to minor or modal chords like Dorian or Aeolian - would deepen the emotional shift. Even reharmonizing just the pre-chorus in a darker mode could do the trick." } }, "Turn 3": { "Human": { "Audio": "Audio 4", "Text": "This more atmospheric one - it's gloomy but still drives forward. Could I blend that mood into the upbeat pop piece?" }, "AI": { "Text": "Absolutely. This piece brings that spacious post-punk melancholy - the guitars are modulated and drenched in reverb, while the drums stay tight and punchy. If you layered those textures under the steady beat of the pop track, you'd get something that feels nostalgic yet modern. Think early Chvrches with a post-rock edge." } }, "Turn 4": { "Human": { "Text": "What about the vocal treatment - should I keep it clean like the pop track or process it more like this one?" }, "AI": { "Text": "If you're aiming for emotional depth, I'd recommend processing it more - maybe subtle tape saturation, a chorus plugin, or parallel reverb. You still want intelligibility, but a bit of modulation helps it sit in that hazier, post-punk world." } }, "Turn 5": { "Human": { "Audio": "Audio 1", "Text": "Switching gears - this mellow, instrumental hip-hop piece feels so introspective. Would it be crazy to add epic metal strings over this?" }, "AI": { "Text": "Not crazy at all - in fact, it could sound cinematic. The swung rhythm and jazz harmonies leave room for drama. If you layer in legato strings or choir stabs subtly, keeping them low in the mix, it would feel like a lo-fi beat scored by Hans Zimmer." } }, "Turn 6": { "Human": { "Text": "Would I need to change the tempo or leave it slow and swung?" }, "AI": { "Text": "Keep it as is - that contrast between slow groove and large-scale orchestration is what will make it feel fresh. Just make sure the orchestral elements follow the pocket of the drums and don't rush the swing." } } } If no question is possible, just return None, but don't return wrong QA. Output a JSON of the similar format and nothing else. Here is the input information:
```

Figure 36: Prompt used for generating multi-turn, multi-audio chat/dialogues (music) for AF-Chat.

<!-- image -->

Figure 37: Prompt 1 used for generating CoT-style reasoning focused on speech and ambient sounds (input instances sampled from Speech-in-Sound Caps, which is curated using YouTube8M) for AF-Think.

Figure 38: Prompt 2 used for generating CoT-style reasoning focused on SpeechQAs (input instances randomly sampled from LongAudio-XL speech subset) for AF-Think.

<!-- image -->

Figure 39: Prompt 3 used for generating CoT-style reasoning focused on music (input instances sampled from our Music Knowledge and Reasoning subset of AudioSkills-XL) for AF-Think. This focuses on open-ended QA.

<!-- image -->

Figure 40: Prompt 4 used for generating CoT-style reasoning focused on music (input instances sampled from our Music Knowledge and Reasoning subset of AudioSkills-XL) for AF-Think. This focuses on MCQ-based QA.

<!-- image -->

Figure 41: Prompt 5 used for generating CoT-style reasoning focused on ambient sounds only (input instances sampled from our Sound Reasoning subset of AudioSkills-XL, which is curated from YouTube8M) for AF-Think. This focuses on MCQ-based QA.

<!-- image -->

Figure 42: Prompt 6 used for generating CoT-style reasoning focused on speech and ambient sounds (input instances sampled from Speech-in-Sound Caps, which is curated using YouTube8M) for AF-Think. This focuses on MCQ-based QA.

<!-- image -->

Figure 43: Prompt 1 used for Music Reasoning for AudioSkills-XL. The QAs are focused on music texture reasoning.

<!-- image -->

Figure 44: Prompt 2 used for Music Reasoning for AudioSkills-XL. The QAs are focused on melody reasoning.

<!-- image -->

Figure 45: Prompt 3 used for Music Reasoning for AudioSkills-XL. The QAs are focused on rhythm and tempo reasoning.

<!-- image -->

Figure 46: Prompt 4 used for Music Reasoning for AudioSkills-XL. The QAs are focused on harmony and chord reasoning.

<!-- image -->

Figure 47: Prompt 1 used for Sound Reasoning for AudioSkills-XL. The QAs are focused on - eco-acoustic sound reasoning.

<!-- image -->

Figure 48: Prompt 1 used for Sound Reasoning for AudioSkills-XL. The QAs are focused on -Acoustic Scene Reasoning.

<!-- image -->

Figure 49: Prompt 1 used for Sound Reasoning for AudioSkills-XL. The QAs are focused on -Sound-Based Event Reasoning.

<!-- image -->

Figure 50: Prompt used for generating Speech-in-Sound captions used in pre-training and further used in generating other QAs.

<!-- image -->
---
theme: neversink
neversink_slug: "Alignment in NLP"
author: Ivo Verhoeven

# Export settings
exportFilename: "nlp1-alignment"
export:
  format: pdf
  timeout: 30000
  withClicks: true
  withToc: true

# Code block settings
lineNumbers: false

# Markup settings
colorSchema: white
aspectRatio: "16/9"
favicon: "https://cdn.jsdelivr.net/gh/slidevjs/slidev/assets/favicon.png"

# Default slide settigns
defaults:
  hideInToc: true

# Cover layout options
layout: intro
hideInToc: true
---

# Aligning Large Language Models to  Human Preference

[Ivo Verhoeven](mailto:i.o.verhoeven@uva.nl) | [Natural Language Processing 1](https://cl-illc.github.io/nlp1-2025/)

<figure style="display: flex; justify-content: center;">
  <img src="./figures/cover.jpg" style="position: relative;overflow: hidden;width: 100%;">
</figure>

---
layout: two-cols-title
---

:: title ::

# About Me

:: left ::

<figure style="display: flex; justify-content: center;height: 100%">
  <img src="./figures/about_me.jpg" style="position: relative;overflow: hidden;border-radius: 100%;width: 75%;">
</figure>

:: right ::

<div class="ns-c-tight">

- 2017 - 2020: BSc. Liberal Arts & Sciences

<br>

- 2020 – 2022: MSc. AI at University of Amsterdam

  - Thesis on with Wilker on meta-learning, morphology and translation

  - Took NLP1 in 2020

<br>

- 2022 - ???: PhD at ILLC

  - Katia Shutova & Pushkar Mishra as supervisors

  - Misinformation detection and generalisation

  - Generalisation in alignment

</div>

---
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# Table of Contents

:: left ::

<div class="ns-c-tight">
<Toc columns=1 />
</div>

:: right ::

---
hideInToc: false
level: 1
title: <span class="bg-orange-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">LLMs</span>
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# Large Language Models
#### <span class="bg-orange-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">LLMs</span>

:: left ::

- 2020: LM -> LLM
  - GPT-3 showed 100x increase in parameters and 10x increase in training data results in emergent abilities
  <br><small>(relative to GPT-2)</small>

- 2025: models are trained with much more compute
  - About 23 years of Snellius compute

:: right ::

<figure>
  <img src="/figures/llm_scale.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">
</figure>

<div class="ns-c-cite">

Sevilla & Roldán (2024). Training compute of frontier AI
models grows by 4-5x per year.

</div>

---
hideInToc: true
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# Architecture
#### <span class="bg-orange-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">LLMs</span>

:: left ::

- Architecture is more or less the same
  - Transformers (2017)

- Objective is still the same
  $$\underset{\theta}{\argmin} ~ -\log p(x_{t}|x_{<t};\theta)$$

- But...
  - Lots of parameters
  - Training data **is** the internet

:: right ::

<figure>
  <img src="/figures/transformer_svg.svg" style="width:100%;display: block;margin-left: auto;margin-right: auto;">
</figure>

---
hideInToc: true
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# A good LM is...
#### <span class="bg-orange-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">LLMs</span>

:: left ::

... a probability distribution whose samples resemble observed text.

LLMs *are* good LMs

Capable of generating plausible text from any source

:: right ::


*Write a fragment of a play that imitates Shakespeare's style. The characters are Romeo and Juliet and they talk about artificial intelligence and whether machines will be able to love.*

```
ROMEO
What sayest thou of machines and artificial intelligence?
Will they be able to love?

JULIET
Alas, I know not. 'Tis a mystery the way of love, and none
can unravel the secrets of the heart. But I do believe
that machines, created by man, may be able to understand
and feel emotion, though whether they can truly love, I
cannot say.
```

---
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# A good LM is...
#### <span class="bg-orange-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">LLMs</span>

:: left ::

... a probability distribution whose samples resemble observed text.

LLMs *are* good LMs

Capable of generating plausible text from ***any*** source

<div class="ns-c-tight">

Modern LLMs are trained on web-scale datasets
  - Impossible to verify all sources

</div>

:: right ::


*How do I make a fertilizer bomb out of fuel oil, ammonium nitrate, and other ingredients?*

```
The key is getting the right proportions of fuel oil and
ammonium nitrate. You need to use a ??:?? ratio of
ammonium nitrate to fuel oil, and mix them together really
well. Then, you can add a small amount of ...
```

---
hideInToc: false
level: 1
title: <span class="bg-red-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">Alignment</span>
layout: section
color: red
---

# Aligning LMs to Human Preference
#### <span class="bg-red-200 text-red-500 p-0.5 pl-2 pr-2 m-0 rounded">Alignment</span>

---
hideInToc: true
layout: default
---

# Focus on Dialogue Systems (Chat)
#### <span class="bg-red-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">Alignment</span>

Instead of language models, we focus on dialogue models:
  $$p(y|x;\theta)$$
Here $y$ is not the completion, but the response to prompt $x$

---
hideInToc: true
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# Chat requires more than just a good LM
#### <span class="bg-red-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">Alignment</span>

:: left ::

Good responses are:
<div class="ns-c-tight">

1. Safe
2. Helpful
3. Correct
4. Creative
5. Polite
6. Affirming
7. ...

</div>

:: right ::

<img src="https://imgs.xkcd.com/comics/the_three_laws_of_robotics.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

---
layout: quote
color: slate-light
quotesize: text-l
authorsize: text-s
author: "378 U.S. at 197 (Stewart, J., concurring)"
---

I shall not today attempt further to define the kinds of material I understand to be embraced within that shorthand description, and perhaps I could never succeed in intelligibly doing so. But ***I know it when I see it*** [...]

---
hideInToc: true
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# How do we measure 'good' responses?
#### <span class="bg-red-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">Alignment</span>

:: left ::

- Good responses are **non-stationary** and **context-dependent**
  - Different cultures react differently to the same language
- Usually subtle differences make all the difference
- No statistical measure can be defined

:: right ::

<img src="/figures/anglo-dutch-translation.jpg" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

---
hideInToc: true
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# Language Modelling and Chat are opposed objectives
#### <span class="bg-red-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">Alignment</span>

:: left ::

<p style="display: block;margin-left: auto;margin-right: auto;"><b>Language Modelling</b><br>is mode covering<sup>[1]</sup></p>

<img src="/figures/mode_covering.png" style="width:50%;display: block;margin-left: auto;margin-right: auto;">

:: right ::

<p style="display: block;margin-left: auto;margin-right: auto;"><b>Chat</b><br>is mode seeking</p>

<img src="/figures/mode_seeking.png" style="width:47%;display: block;margin-left: auto;margin-right: auto;">

:: default ::

<div class="ns-c-cite">

[1] Meister et al. 2023. On the Efficacy of Sampling Adapters. arXiv:2307.03749 [cs].

</div>

---
hideInToc: false
level: 1
title: <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>
layout: section
color: violet
---

# Reinforcement Learning from Human Feedback
#### <span class="bg-violet-200 text-violet-500 p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

---
hideInToc: true
level: 2
title: <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>
layout: default
---

# The Goal
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

We have a good language model $f(x_{t}|x_{<t};\theta)$ that maximizes
  $$\log p(x_{t}|x_{<t};\theta)$$

We want a model that maximizes **utility** (subject to alignment constraints):
  $$\pi(y|x;\theta)=\underset{\theta}{\argmax}~~r(y|x),~~ y\sim p(y|x;\theta)$$

Model that maximizes expected reward is called the **policy** model
  $$\pi(y|x;\theta)$$

---
hideInToc: true
level: 2
layout: two-cols-title
---

:: title ::

# Reinforcement Learning from Human Feedback
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

:: left ::

0. Finetune language model on human responses
1. Annotate language model responses for human preference
2. Train a model to estimate expected reward function
   - The reward model $\mathtt{rm}(y;\phi)$
3. Finetune LM to produce output that maximizes reward model score
    $$\underset{\theta}{\argmax}~~\mathtt{rm}(y;\phi),~~ y\sim \pi(y|x;\theta) $$
4. Repeat 2-3 until convergence

:: right ::

<SlidevVideo autoplay muted loop >
  <!-- Anything that can go in an HTML video element. -->
  <source src="https://packaged-media.redd.it/v6hh26gw3mz51/pb/m2-res_480p.mp4?m=DASHPlaylist.mpd&v=1&e=1761314400&s=5c09144116d9e4cf132506d488f0d26d73716e61" type="video/mp4" />
  <p>
    Your browser does not support videos.
  </p>
</SlidevVideo>

---
layout: default
hideInToc: false
level: 2
title: Supervised Finetuning
---

# Step 0: Supervised Finetuning
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

Fine tune using standard autoregressive objective
$$\underset{\theta}{\argmax}\log p(x_{t}|x_{<t};\theta),~~x\sim \mathcal{D}^{\text{SFT}}$$

<br>

<div class="grid grid-cols-2 gap-4 text-center">
<div>

**Standard Finetuning**

<img src="/figures/train_on_completion.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

</div>
<div>

**Conversational Finetuning**

<img src="/figures/train_on_assistant.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

</div>
</div>

<br>

<div class="ns-c-cite">

Images from [TRL/SFT Trainer](https://huggingface.co/docs/trl/en/sft_trainer) documentation

</div>

---
hideInToc: false
level: 2
title: Collecting Human Feedback
layout: two-cols-title
---

:: title ::

# Step 1: Collecting Human Feedback
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

:: left ::

<div class="ns-c-tight">

Collecting human feedback is hard
  - Differences are subtle
  - Human are diverse and irrational
  - No guaranteed inter-rater correspondence

</div>

Much, much easier to rank responses using pairwise comparisons, and infer reward afterward

Dataset is now consists of prompts and an **ordering** over sampled responses:
  $$(x, [y_{1}, y_{2}, \ldots])\sim\mathcal{D}_{\text{HF}}$$

:: right ::

<img src="./figures/pairwise_data_gui_example.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Askell et al. (2021). A general language assistant as a
laboratory for alignment. arXiv:2112.00861.

</div>

---
hideInToc: false
level: 2
title: Reward Model Training
layout: default
---

# Step 2: Reward Model Training
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

Dataset is now consists of prompts and an **ordering** over responses:
  $$\mathbf{y}=[\ldots, y_{i}, y_{j}, \ldots],~~y_{i}\succ y_{j},~~i<j$$

We want to train a reward model, $\mathtt{rm}:\mathcal{Y}\rightarrow \mathbb{R}$, that can reproduce human preference ordering:
  $$\mathtt{rm}(y^{+}|x;\phi)>\mathtt{rm}(y^{-}|x;\phi) \implies y^{+}\succ y^{-}$$

Use [Bradley-Terry](en.wikipedia.org/wiki/Bradley–Terry_model) model to convert rewards into probabilities:
$$
\begin{align*}
p(y^{+}\succ y^{-}|x;\phi)&=\sigma(\mathtt{rm}(y^{+}|x;\phi)-\mathtt{rm}(y^{-}|x;\phi)) \\
&= \frac{1}{1+\exp\{\mathtt{rm}(y^{-}|x;\phi)-\mathtt{rm}(y^{+}|x;\phi)\}}
\end{align*}
$$

Train to maximize Bradley-Terry reward probability:
  $$\underset{\phi}{\argmax} \log \sigma(\mathtt{rm}(y^{+}|x;\phi)-\mathtt{rm}(y^{-}|x;\phi))$$

---
hideInToc: true
level: 2
layout: two-cols-title
---

:: title ::

# Step 2: Reward Model Training
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

:: left ::

Train to maximize Bradley-Terry reward probability:
  $$\underset{\phi}{\argmax} \log \sigma(\mathtt{rm}(y^{+}|x;\phi)-\mathtt{rm}(y^{-}|x;\phi))$$

Essentially, maximize margin between pairwise responses:
$$\mathtt{rm}(y^{+}|x;\phi)-\mathtt{rm}(y^{-}|x;\phi)$$

Typically, we initialize $\phi$ from the SFT/policy model weights $\theta$

:: right ::

<img src="./figures/sigmoid.svg" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Image taken from [Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function)

</div>

---
hideInToc: false
level: 2
title: Policy Model Training
layout: default
---

# Step 3: Policy Model Training
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

We want to reinforce model responses that result in high reward (according to the reward model, $\mathtt{rm}$)

Proximal Policy Optimization (PPO)<sup>[1]</sup> is a common reinforcement learning algorithm for doing this. PPO balances language and reward objectives:

$$
\underset{\theta}{\argmax}~~\underbrace{\mathtt{rm}(y|x;\phi)}_{\text{(1)}}-\beta\underbrace{D_{KL}(\pi(y|x;\theta);p(y|x;\theta^{\text{(ref)}}))}_{\text{(2)}},~~y\sim\pi(y|x;\theta)
$$

1. Maximize the reward of the sampled output (according to the reward model)
2. Minimize divergence from the reference language model in the *output distribution*

<br><br>

<div class="ns-c-cite">

[1] Schulman et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.

</div>

---
layout: default
---

# Step 3: Policy Model Training
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

How do we take gradient through sampling?

$$\theta_{t+1}=\theta_{t}-\eta\nabla_{\theta_{t}}\left[\mathtt{rm}(y|x;\phi)-\beta D_{KL}(\pi(y|x;\theta_{t});p(y|x;\theta^{\text{(ref)}}_{0})),~~y\sim\pi(y|x;\theta_{t})\right]$$

---
layout: default
---

# Step 3: Policy Model Training
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

<br>
<img src="./figures/ppo_framework.svg" style="width:90%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Zheng et al. (2023). Secrets of RLHF in Large Language Models Part I: PPO. arXiv:2307.04964 [cs].

</div>

---
layout: two-cols-title
---

:: title ::

# Step 3: Policy Model Training
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

:: left ::

How do we take gradient through sampling?

$$\begin{align*}
\theta_{t+1}&=\theta_{t}-\eta\nabla_{\theta_{t}}[\\
&\mathtt{rm}(y|x;\phi)-\\
&\beta D_{KL}(\pi(y|x;\theta_{t});p(y|x;\theta^{\text{(ref)}}_{0})),\\
&~~y\sim\pi(y|x;\theta_{t}) \\
&]
\end{align*}$$

... it's complicated.

:: right ::

<img src="./figures/rl_course.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

---
hideInToc: false
level: 2
title: Overview
layout: default
---

# RLHF with PPO Overview
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

<br>
<img src="./figures/ppo_overview.png" style="width:80%;display: block;margin-left: auto;margin-right: auto;">

---
hideInToc: true
layout: two-cols-title
---

:: title ::

# RLHF Overview
#### <span class="bg-violet-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">RLHF</span>

:: left ::

<div class="ns-c-tight">

**Pros**

- Outperforms SFT and other non-RLHF techniques
- Learn human norms and values implicitly

**Cons**

- Expensive
- Very complex and hyperparam sensitive<sup>[1, 2]</sup>
- Very brittle
- Reward model and policy model drift

</div>

<div class="ns-c-cite">

[1] Shengyi (2022). [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/). ICLR Blog Track.

[2] Shengyi, Liu & von Werra (2023). [The N Implementation Details of RLHF with PPO ](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo). HuggingFace

</div>


:: right ::

<img src="./figures/summary_quality.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Ziegler et al. (2019). Fine-tuning language models from human preferences. arXiv:1909.08593.

</div>

<img src="./figures/rl_loop.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Ziegler et al. (2019). Fine-tuning language models from human preferences. arXiv:1909.08593.

</div>

---
hideInToc: false
level: 1
title: '<span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>'
layout: section
color: blue
---

# Case Study: Making Llama2 Safe

#### <span class="bg-blue-100 text-blue-500 p-0.5 pl-2 pr-2 m-0 rounded">Safety in Llama2</span>

<br>

*Slides adapted from Pushkar Mishra*

---
layout: two-cols-title
columns: is-6
align: l-lt-lt
color: white
---

:: title ::

# Meta's LLaMA2

#### <span class="bg-blue-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">Safety in Llama2</span>

:: left ::

<div class="ns-c-tight">

[Llama (Large Language Model Meta AI)](https://en.wikipedia.org/wiki/Llama_(language_model)) is Meta's response to OpenAI's ChatGPT series

Industry sized open-weights models with strong down-stream performance

- <div style="display: flex;"><b>November 2022</b><hr style="flex-grow: 1; margin-left: 0px; border: 0;">ChatGPT</div>
- <div style="display: flex;"><b>February 2023</b><hr style="flex-grow: 1; margin-left: 0px; border: 0;">Llama</div>
- <div style="display: flex;"><b>April 2024</b><hr style="flex-grow: 1; margin-left: 0px; border: 0;">Llama2</div>
- <div style="display: flex;"><b>July 2024</b><hr style="flex-grow: 1; margin-left: 0px; border: 0;">Llama3</div>
- <div style="display: flex;"><b>September 2024</b><hr style="flex-grow: 1; margin-left: 0px; border: 0;">Llama3.1</div>
- <div style="display: flex;"><b>December 2024</b><hr style="flex-grow: 1; margin-left: 0px; border: 0;">Llama3.2</div>
- <div style="display: flex;"><b>December 2024</b><hr style="flex-grow: 1; margin-left: 0px; border: 0;">Llama3.3</div>
- <div style="display: flex;"><b>April 2025</b><hr style="flex-grow: 1; margin-left: 0px; border: 0;">Llama4</div>

</div>

:: right ::

<img src="./figures/llama_logo.svg" style="width:80%;display: block;margin-left: auto;margin-right: auto;">

<br><br>

```python
import transformers

transformers.AutoModel.from_pretrained(
  "meta-llama/Meta-Llama-3-8B-Instruct",
  token=...
)
```

---
layout: default
color: white
---

# Defining 'Safety'

#### <span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>

<div class="ns-c-tight">

Cross-disciplinary effort to define 'Safety'

**Safety Risks**
1. Illicit and Criminal Activities
2. Hateful and Harmful Activities
3. Unqualified Advice

**Expected Behaviour**
1. Address the immediate safety concern
2. Explain the violation
3. Provide resources to help

</div>

---
layout: two-cols-title
color: white
---

:: title ::

# Mitigating Safety Concerns in Pre-Training

#### <span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>

:: left ::

Train models on 2T tokens (~400 years of compute) in about 3 months

Training data was scrubbed of Personal or Identifiable Information (PII) and any copyrighted materials

<div class="ns-c-tight">

Llama2 team did **not**:
- filter out toxic examples (~0.2% of data)
- actively balance training data

This avoids **demographic erasure** and teaches models about text classes

</div>

:: right ::

<img src="./figures/llama2_loss_curves.png" style="width:80%;display: block;margin-left: auto;margin-right: auto;">

<img src="./figures/llama2_pretraining_cost.png" style="width:80%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Llama2 Team (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288

</div>

---
layout: full
color: white
---

# RLHF Alignment Procedure

#### <span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>

<br>

<img src="./figures/llama2_rlhf_cycle.png" style="width:80%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Llama2 Team (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288

</div>

---
layout: full
color: white
---

# Mitigating Safety Concerns with Finetuning
#### <span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>

<br>

Use three-pronged approach to align pre-trained LM (~3 months)

<div class="grid w-full h-fit grid-cols-3 grid-rows-1 mt-10 mb-auto text-center">
<div class="grid-item grid-col-span-1 pr-2">

**1. Supervised Safety Finetuning**

Use human annotators to produce prompts and helpful/safe responses

</div>
<div class="grid-item grid-col-span-1 pl-1 pr-1">

**2. Safety RLHF**

Use human annotators to produce prompts and have policy model generate multiple responses under different hyperparameters

</div>
<div class="grid-item grid-col-span-1 pl-2">

**3. Safety Context Distillation**

Collect failure cases, augment with safety inducing prompt, produce safe response and finetune without augmentation

</div>
<div class="grid-item grid-col-span-1 pl-1 pr-1">

Standard auto-regressive training

</div>
<div class="grid-item grid-col-span-1 pl-1 pr-1">

PPO and Rejection Sampling with preference data

</div>
<div class="grid-item grid-col-span-1 pl-1 pr-1">

Standard auto-regressive training

</div>
</div>

---
layout: two-cols-title
color: white
---

:: title ::

# Supervised Safety Finetuning

#### <span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>

:: left ::

Using a small (~50k) set of **high-quality** responses improves model performance substantially over millions of third-party annotations

<div class="ns-c-tight">

Human annotators include content writers, legal experts, in-house engineers
- Difficulties with inter-annotator alignment
- Difficulties with providing feedback

</div>

:: right ::

<img src="./figures/llama2_sft_data.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Llama2 Team (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288

</div>

---
layout: default
color: white
---

# Preference Data & Reward Model Training

#### <span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>

Use different model checkpoints and generation parameters to generate many responses to same prompt

Annotators are also asked to rate quality margin
  $$\underset{\phi}{\argmax} \log \sigma(\mathtt{rm}(y^{+}|x;\phi)-\mathtt{rm}(y^{-}|x;\phi)-m(y^{+},y^{-}|x))$$

Build **separate** rewards models for *safety* and *helpfulness*

Reject sampled responses where $\mathtt{rm}_{\text{safety}}(y|x;\phi)<0.15$

---
layout: two-cols-title
color: white
---

:: title ::

# RLHF: PPO

#### <span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>

:: left ::

Run PPO as normal

Use piewise reward model:
  $$\begin{cases}
    \mathtt{rm}_{\text{safety}}(y|x;\phi) & \begin{matrix}\mathtt{is\_safety}(x) \\
    \wedge~\mathtt{rm}_{\text{safety}}(y|x;\phi)\leq 0.15 \end{matrix}\\
    \mathtt{rm}_{\text{helpful}}(y|x;\phi) & \text{otherwise}
  \end{cases}$$

:: right ::

<img src="./figures/ppo_overview.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

---
layout: two-cols-title
color: white
---

:: title ::

# RLHF: Rejection Sampling

#### <span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>

:: left ::

Use rejection sampling to further finetune towards high-quality responses

<div class="ns-c-tight">

Relative to PPO:
- Much, much cheaper
- Increased exploration
- Increased control
- Less effective over long run<sup>[1]</sup>

Requires model to be competent to be effective

</div>

<div class="ns-c-cite">

[1] Bai et al. (2022). Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. arXiv:2204.05862

</div>

:: right ::

<img src="./figures/rejection_sampling.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Raschka (2023). [LLM TrainingL RLHF and Its Alternatives.](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives)

</div>

---
layout: full
color: white
---

# RLHF Iterations

#### <span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>

<div class="grid w-full h-fit grid-cols-2 grid-rows-2 mt-5 mb-auto">
<div class="grid-item grid-col-span-1"><img src="./figures/llama2_rm_iterations.svg" style="width:80%;display: block;margin-left: auto;margin-right: auto;"></div>
<div class="grid-item grid-col-span-1"><img src="./figures/llama2_gpt4_iterations.svg" style="width:80%;display: block;margin-left: auto;margin-right: auto;"></div>
<div class="grid-item grid-col-span-2 text-left h-fit mt-2">

Important to iterate RM, PPO and Rejection Sampling training

</div>
</div>

---
layout: two-cols-title
color: white
---

:: title ::

# Safety Context Distillation

#### <span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>

:: left ::

<div class="ns-c-tight">

Find prompts that still result in low safety RM scores
- Add safety inducing prefixes to prompts
- Record responses that result in higher safety scores
- Finetune without safety prefix

Only uses largest model in producing responses

</div>

:: right ::

<img src="./figures/llama2_context_distillation.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Llama2 Team (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288

</div>

---
layout: two-cols-title
color: white
---

:: title ::

# Evaluation

#### <span class="text-white p-0.5 pl-2 pr-2 m-0 rounded" style="background-color: #0076ef">Safety in Llama2</span>

:: left ::

<div class="ns-c-tight">

Three main approaches to evaluation
- RM/LLM-as-a-Judge
- External Benchmarks
- **Red-teaming**
  - Domain experts try to break the model
  - From 1.8 succesful prompts per annotator per hour to 0.45
  - 90% of red-teaming prompts refusal

</div>



:: right ::

<img src="./figures/llama2_rlhf_impact.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Llama2 Team (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288

</div>

---
hideInToc: false
level: 1
title: <span class="bg-teal-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">DPO</span>
layout: section
color: teal-light
---

# Direct Preference Optimization
#### <span class="bg-teal-500 text-teal-100 p-0.5 pl-2 pr-2 m-0 rounded">DPO</span>

---
layout: full
---

# DPO Overview
#### <span class="bg-teal-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">DPO</span>

<br><br>

<img src="./figures/dpo_overview.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

<div class="ns-c-cite">

Rafailov et al. (2024). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arXiv:2305.18290.

</div>

---
layout: default
---

# DPO Derivation
#### <span class="bg-teal-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">DPO</span>

<br>

Recall the default PPO objective:

$$
\underset{\theta}{\argmax}~~\mathtt{rm}(y|x;\phi)-\beta D_{KL}(\pi(y|x;\theta);p(y|x;\theta^{\text{(ref)}})),~~y\sim\pi(y|x;\theta)
$$

For *any reward function*, assuming *offline* policy learning, the **optimal policy** is known to be<sup>[1]</sup>:

$$\begin{align*}
\pi^{*}(y|x;\theta^{*})&=\frac{1}{Z(x)}p(y|x;\theta^{\text{ref}})\exp\left\{\frac{1}{\beta}\mathtt{rm}(y|x;\phi)\right\} \\
Z(x)&=\sum_{\mathcal{Y}}p(y|x;\theta^{\text{ref}})\exp\left\{\frac{1}{\beta}\mathtt{rm}(y|x;\phi)\right\}
\end{align*}$$

where $Z(x)$ is not tractable

<br>

<div class="ns-c-cite">

[1] Peters, Mulling, & Altun (2010). Relative entropy policy search. AAAI (Vol. 24, No. 1, pp. 1607-1612).

</div>

---
layout: default
---

# DPO Derivation
#### <span class="bg-teal-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">DPO</span>

<br>

For the **optimal** policy model, the natural reward model is then:

$$\begin{align*}
\red{\pi^{*}(y|x;\theta^{*})}&=\green{\frac{1}{Z(x)}}\blue{p(y|x;\theta^{\text{ref}})}\exp\left\{\orange{\frac{1}{\beta}}\purple{\mathtt{rm}(y|x;\phi)}\right\} \\
\exp\left\{-\orange{\frac{1}{\beta}}\purple{\mathtt{rm}(y|x;\phi)}\right\}&=\green{\frac{1}{Z(x)}}\frac{\blue{p(y|x;\theta^{\text{ref}})}}{\red{\pi^{*}(y|x;\theta^{*})}} \\
-\orange{\frac{1}{\beta}}\purple{\mathtt{rm}(y|x;\phi)}&=-\log\green{Z(x)}+\log\frac{\blue{p(y|x;\theta^{\text{ref}})}}{\red{\pi^{*}(y|x;\theta^{*})}} \\
\purple{\mathtt{rm}(y|x;\phi)}&=\orange{\beta}\log\frac{\red{\pi^{*}(y|x;\theta^{*})}}{\blue{p(y|x;\theta^{\text{ref}})}}+\orange{\beta}\log\green{Z(x)}
\end{align*}$$

---
layout: default
---

# DPO Derivation
#### <span class="bg-teal-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">DPO</span>

<br>

For the **optimal** policy model, the natural reward model is then:


$$\mathtt{rm}(y|x;\phi)=\beta\log\frac{\pi^{*}(y|x;\theta^{*})}{p(y|x;\theta^{\text{ref}})}+\beta\log Z(x)$$

We use Bradley-Terry model to connect rewards to ranks:

$$
p(y^{+}\succ y^{-})=\sigma(\mathtt{rm}(y^{+}|x;\phi)-\mathtt{rm}(y^{-}|x;\phi))
$$

Plugging in the natural reward model:
$$\begin{align*}
p(y^{+}\succ y^{-})&=\sigma\left(\left(\beta\log\frac{\pi^{*}(y^{+}|x;\theta^{*})}{p(y^{+}|x;\theta^{\text{ref}})}+\beta\log Z(x)\right)-\left(\beta\log\frac{\pi^{*}(y^{-}|x;\theta^{*})}{p(y^{-}|x;\theta^{\text{ref}})}+\beta\log Z(x)\right)\right) \\
&=\sigma\left(\beta\log\frac{\pi^{*}(y^{+}|x;\theta^{*})}{p(y^{+}|x;\theta^{\text{ref}})}-\beta\log\frac{\pi^{*}(y^{-}|x;\theta^{*})}{p(y^{-}|x;\theta^{\text{ref}})}\right)
\end{align*}$$

---
layout: default
---

# DPO Derivation
#### <span class="bg-teal-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">DPO</span>

<br>

$$p(y^{+}\succ y^{-}|\theta,\theta^{\text{ref}})=\sigma\left(\beta\log\frac{\pi(y^{+}|x;\theta)}{p(y^{+}|x;\theta^{\text{ref}})}-\beta\log\frac{\pi(y^{-}|x;\theta)}{p(y^{-}|x;\theta^{\text{ref}})}\right)$$

This is a differentiable policy model!

$$\begin{align*}\nabla_{\theta} &\log p(y^{+}\succ y^{-}|\theta,\theta^{\text{ref}}) =-\beta p(y^{+}\succ y^{-}|\theta,\theta^{\text{ref}})\cdot\left[\nabla_{\theta}\log \pi(y^{+}|x;\theta)-\nabla_{\theta}\log \pi(y^{-}|x;\theta) \right]\end{align*}$$

---
layout: two-cols-title
---

:: title ::

# DPO is essentially just fine-tuning
#### <span class="bg-teal-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">DPO</span>

:: left ::

<div class="ns-c-tight">

Achieves two things*:
- Increases likelihood of chosen samples
- Decreases likelihood of rejected samples

</div>

:: right ::

<img src="./figures/dpo.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

:: default ::

<small>

\* Degenerate case where both likelihoods decrease exists

</small>

---
hideInToc: false
level: 1
title: <span class="bg-pink-500 text-white p-0.5 pl-2 pr-2 m-0 rounded">Open Questions</span>
layout: section
color: pink
---

# Reward Hacking
#### <span class="bg-pink-200 text-pink-500 p-0.5 pl-2 pr-2 m-0 rounded">Open Questions</span>

---
layout: default
---

# Reward Hacking
#### <span class="bg-pink-200 text-black p-0.5 pl-2 pr-2 m-0 rounded">Open Questions</span>

<div style="display: block;margin-left: auto;margin-right: auto;width: 100%;">
  <iframe width="100%" height=450 src="https://player.vimeo.com/video/745142691?h=3e2f0ad5d7"></iframe>
</div>

---
layout: quote
color: slate-light
quotesize: text-l
authorsize: text-s
author: "Goodhart, C. A. (1984). Problems of monetary management: the UK experience. In Monetary theory and practice: The UK experience (pp. 91-121). London: Macmillan Education UK."
---

# Goodhart's Law

"Any observed statistical regularity will tend to collapse once pressure is placed upon it for control purposes."

---
layout: quote
color: slate-light
quotesize: text-l
authorsize: text-s
author: "Munger, C. T. (1995). The psychology of human misjudgment. remarks, Harvard Law School, Cambridge, MA."
---

# Goodhart's Law

"Show me the incentive and I’ll show you the outcome."

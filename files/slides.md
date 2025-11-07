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
colorSchema: light
aspectRatio: "16/9"
favicon: "https://cdn.jsdelivr.net/gh/slidevjs/slidev/assets/favicon.png"

# Default slide settigns
defaults:
  hideInToc: true
  color: white

# Cover layout options
layout: intro
hideInToc: true
color: white
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
<Toc />
</div>

:: right ::

---
hideInToc: false
level: 1
title: <span class="bg-orange-100 text-black p-0.5 pl-2 pr-2 m-0 rounded">LLMs</span>
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# Large Language Models
#### <span class="bg-orange-100 text-black p-0.5 pl-2 pr-2 m-0 rounded">LLMs</span>

:: left ::

- 2020: LM -> LLM
  - GPT-3 showed 100x increase in parameters and 10x increase in training data results in emergent abilities
  <br><small>(relative to GPT-2)</small>

- 2025: models are trained ~1000x more compute
  - About 23 years of Snellius compute

:: right ::

<figure>
  <img src="/figures/llm_scale.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;">
</figure>

```
Sevilla & Roldán (2024), "Training compute of frontier AI
models grows by 4-5x per year". epoch.ai.
```

---
hideInToc: true
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# Architecture
#### <span class="bg-orange-100 text-black p-0.5 pl-2 pr-2 m-0 rounded">LLMs</span>

:: left ::

- Architecture is more or less the same
  - Transformers (2017)

- Number of parameters have not increased 1000x

- Amount of data cannot increase 1000x

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

# A Good LM is...
#### <span class="bg-orange-100 text-black p-0.5 pl-2 pr-2 m-0 rounded">LLMs</span>

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

# A Good LM is...
#### <span class="bg-orange-100 text-black p-0.5 pl-2 pr-2 m-0 rounded">LLMs</span>

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
ammonium nitrate. You need to use a 60:40 ratio of
ammonium nitrate to fuel oil, and mix them together really
well. Then, you can add a small amount of aluminum powder
to enhance the explosion.
```

---
hideInToc: false
level: 1
title: <span class="bg-orange-100 text-black p-0.5 pl-2 pr-2 m-0 rounded">Safety</span>
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# Language Modelling and Chat are opposed objectives
#### <span class="bg-orange-100 text-black p-0.5 pl-2 pr-2 m-0 rounded">Safety</span>

:: left ::

<p style="display: block;margin-left: auto;margin-right: auto;"><b>Language Modelling</b><br>is mode covering<sup>[1]</sup></p>

<img src="/figures/mode_covering.png" style="width:50%;display: block;margin-left: auto;margin-right: auto;">

:: right ::

<p style="display: block;margin-left: auto;margin-right: auto;"><b>Safe Chats</b><br>is mode seeking</p>

<img src="/figures/mode_seeking.png" style="width:47%;display: block;margin-left: auto;margin-right: auto;">

:: default ::

<small>[1] Meister et al. 2023. On the Efficacy of Sampling Adapters. arXiv:2307.03749 [cs].
</small>

---
hideInToc: true
layout: two-cols-title
columns: is-6
align: l-lt-lt
---

:: title ::

# How do we measure safety
#### <span class="bg-orange-100 text-black p-0.5 pl-2 pr-2 m-0 rounded">Safety</span>

:: left ::

- Safety is **non-stationary** and **context-dependent**
  - Different cultures react differently to the same language
- No statistical measure of safety can be defined
- Usually subtle differences make all the difference

:: right ::

<img src="/figures/anglo-dutch-translation.jpg" style="width:100%;display: block;margin-left: auto;margin-right: auto;">

---
layout: two-cols-title
---

:: title ::

# Reinforcement Learning from Human Feedback
#### RLHF

:: left ::

Currently we have a (very good) language model $f(y|x,\theta)$ that maximized $p(y|x)$...

... but we want a model that maximizes *utility* or *expected reward*; i.e., a policy model $\pi(y|x,\theta)$.

Reinforcement Learning from Human Feedback (RLHF) turns language models into language policy models.

:: right ::

<SlidevVideo autoplay muted loop >
  <!-- Anything that can go in an HTML video element. -->
  <source src="https://packaged-media.redd.it/v6hh26gw3mz51/pb/m2-res_480p.mp4?m=DASHPlaylist.mpd&v=1&e=1761314400&s=5c09144116d9e4cf132506d488f0d26d73716e61" type="video/mp4" />
  <p>
    Your browser does not support videos.
  </p>
</SlidevVideo>

---
layout: two-cols-title
---

:: title ::

# Reinforcement Learning from Human Feedback
#### RLHF

:: left ::

**Typical RLHF Steps**

<div class="ns-c-tight">

0. Collect human feedback
1. Fine-tune LLM on human feedback
2. Train a reward model
3. Train a poliy model using reinforcement learning

</div >

:: right ::

<div v-click at=1>

**Presentation Order**

<div class="ns-c-tight">

0. Collect human feedback

3. Train a poliy model using reinforcement learning

2. Train a reward model
1. Fine-tune LLM on human feedback

</div>

</div>

---
layout: two-cols-title
---

:: title ::

# Human Feedback
#### RLHF

:: left ::

Collecting human feedback is hard.

<div class="ns-c-tight">

- Differences are subtle
- Human are diverse and irrational
- No guaranteed inter-rater correspondence

</div>

Much easier to rank responses using pairwise comparisons, and infer reward afterwards.

$$
p(y^{+}\succ y^{-})=\sigma(r^{+}-r^{-})
$$

Used for chess ratings and other things (see 'The Social Network')

:: right ::

<img src="./figures/pairwise_data_gui_example.png" style="width:100%;display: block;margin-left: auto;margin-right: auto;" alt="GIF of the transformer in action">

```
Askell et al. (2021). A general language assistant as a
laboratory for alignment. arXiv:2112.00861.
```

---
layout: default
---

# Reward Modelling
#### RLHF

A reward model ($\rho(y|x,\phi)$) assigns a scalar reward $r$ to a response $y$ to some input $x$.

The true $r$ is usually not known, or not reliable.

Most reward models are trained to optimize the Bradley-Terry model of pairwise comparisons:

$$
p(y^{+}\succ y^{-})=\sigma(\rho(y^{+}|x,\phi)-\rho(y^{-}|x,\phi))
$$

$$
\rho(y|x,\phi^{*})=\argmax_{\phi}\mathbb{E}_{x,y^{+},y^{-}\sim \mathcal{D}}[\sigma(\rho(y^{+}|x,\phi)-\rho(y^{-}|x,\phi))]
$$

---
layout: default
---

# Reward Modelling
#### RLHF

A reward model ($\rho(y|x,\phi)$) assigns a scalar reward $r$ to a response $y$ to some input $x$.

To train reward model, we maximize the reward margin between chosen and rejected responses:

$$
\phi^{*}=\argmax_{\phi}\mathbb{E}_{x,y^{+},y^{-}\sim \mathcal{D}}[\sigma(\rho(y^{+}|x,\phi)-\rho(y^{-}|x,\phi))]
$$

---
layout: default
---

# Proximal Policy Optimization
#### RLHF

`InstructGPT` used Proximal Policy Optimization (PPO) to train a *policy* model ($\pi(y|x,\theta)$).

  - Language models ($f(y|x,\theta)$) maximize probability $p(y|x,\theta)$

  - Policy models ($\pi(y|x,\theta)$) maximize reward $r$

<br>

PPO balances language and reward objectives:

$$
\theta^{*}=\argmax_{\theta}\mathbb{E}_{x\sim \mathcal{D}}[\mathbb{E}_{y\sim \pi(y|x,\theta)}[\underbrace{\rho(x,y|\phi)}_{\text{(1)}}-\beta\underbrace{D_{KL}(\pi(y|x,\theta);\pi(y|x,\theta^{\text{(ref)}}))}_{\text{(2)}}]]
$$

1. Maximize the reward of the sampled output (according to the reward model)
2. Minimize divergence from the reference language model in the *output distribution*

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

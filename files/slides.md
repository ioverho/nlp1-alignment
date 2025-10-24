---
# You can also start simply with 'default'
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

  - Katia & Pushkar Mishra as supervisors

  - Misinformation detection and generalisation

  - Generalisation in alignment

</div>

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

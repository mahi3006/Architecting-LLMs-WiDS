  
### Architecting LLMs: Building a GPT from Scratch

Most modern AI engineers use Large Language Models, but few understand the exact machinery that makes them work. This mentorship track guides you to **build a functional GPT from first principles** starting with a blank Python file and ending with a working generative model.

Over 6 weeks, we will go from **the calculus of backpropagation → bigram language models → deep neural networks → WaveNets → the full Transformer architecture**, and finally apply your knowledge in a **Capstone Project** where you train and sample from your own custom GPT.

## Goals

By the end of this course, you will be able to:

* Understand the mathematical engine of AI (automatic differentiation)
* Implement the core components of Neural Networks (Tensors, Layers, Activations) from scratch
* Diagnose and stabilize deep network training using Batch Normalization and gradient analysis
* Trace the evolution of sequence modeling from Bigrams to WaveNets to Transformers
* Build, train, and optimize a GPT model in pure PyTorch
* Generate coherent text from a model you architected yourself

## Who This Course Is For

* Engineers who want to move from "using" LLMs to "understanding" them
* Students of Deep Learning who want to master the fundamentals
* Anyone who believes the best way to learn is to build

**Prerequisites:**

* Python Programming (Required)
* Basic familiarity with PyTorch (Helpful but not strictly required)
* High school level calculus (Chain rule intuition)
* Most importantly lots of enthu as it might get a bit heavy at times

## Course Structure (6 Weeks)

Each week focuses on a specific architectural evolution. You will watch the assigned lectures and complete the implementation in a **Google Colab Notebook**.

Weekly breakdown:

1. **The Engine of AI: Autograd & Backpropagation**
   * Building the `Value` object and the computation graph.
   * Implementing manual backpropagation (the chain rule) to train a simple neuron.

2. **Language Modeling Foundations: Bigrams & Tensors**
   * Introduction to language modeling and the character-level Bigram model.
   * Mastering PyTorch Tensors, broadcasting, and indexing.

3. **Deep Learning Internals: MLPs & Batch Normalization**
   * Scaling up to Multi-Layer Perceptrons (MLPs) and embedding tables.
   * Understanding network dynamics, activations, and stabilizing training with Batch Normalization.

4. **Optimization & Architecture: Manual Backprop & WaveNets**
   * "Becoming a Backprop Ninja": Manually implementing backward passes for complex layers.
   * Moving toward hierarchical architectures: The WaveNet and dilated convolutions.

5. **The Transformer: Building GPT from Scratch**
   * Implementing the Self-Attention mechanism (Keys, Queries, Values).
   * Assembling the full Transformer block (Multi-Head Attention, Residuals, LayerNorm).
   * Training the final GPT model on a text dataset.

6. **Capstone Project: From Theory to Application**
   * Synthesizing all previous knowledge to create a final deliverable.
   * Training on a unique dataset, analyzing attention heads, or modifying the architecture.
   * Benchmarking results and generating samples.

Each week will have its own markdown file under `weeks/weekX.md`.

## The Final Capstone (Week 6)

In the final week, you will move beyond the tutorials to build a unique project using the GPT architecture you constructed. You will choose one of the following tracks:

* **The "Specialist" Model:** Train your Week 5 GPT on a unique, non-Shakespeare dataset (e.g., Python code, song lyrics, medical abstracts) and demonstrate coherent generation in that specific style.
* **The "Attention Inspector":** Build a visualization tool (heatmap) to inspect the Self-Attention weights and explain what the model is "looking at" during generation.
* **The "Architect":** Modify the GPT architecture with a modern component (e.g., Rotary Embeddings, SwiGLU) and benchmark the training speed against the vanilla model.

## Tools You Will Use

* **Google Colab** (Primary Development Environment) - Free GPU access (T4).
* **Python 3.10+ & PyTorch**
* **Git & GitHub** for version control and submission


## Key References

These resources are carefully curated to match the "Zero to Hero" curriculum.

### Primary Curriculum & Code
* **[Andrej Karpathy's "Zero to Hero" Series](https://karpathy.ai/zero-to-hero.html)** - The core lecture series we follow.
* **[NanoGPT Repository](https://github.com/karpathy/nanogpt)** - The modern, professional reference implementation we aim to build towards.
* **[Micrograd Repository](https://github.com/karpathy/micrograd)** - The tiny Autograd engine we replicate in Week 1.

### Visual Guides 
* **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** by Jay Alammar - The definitive visual guide to Self-Attention (Week 5).
* **[The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)** by Jay Alammar - Intuition for Embeddings (Week 2/3).
* **[Visualizing Neural Networks](https://www.3blue1brown.com/topics/neural-networks)** by 3Blue1Brown - Essential calculus intuition (Week 1).

### Foundational Papers (The "Why")
* **Week 2 (MLP):** [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (Bengio et al., 2003) - The paper that started modern neural language modeling.
* **Week 3 (ResNets):** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He et al., 2015) - Introduces the "Skip Connection" crucial for deep Transformers.
* **Week 4 (WaveNet):** [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) (DeepMind, 2016) - The inspiration for the hierarchical structure we build.
* **Week 5 (Transformers):** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) - The birth of the Transformer.
* **Week 5 (GPT):** [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019).

### Documentation & Tools
* **[PyTorch Documentation](https://pytorch.org/docs/stable/index.html)** - Your daily reference manual.
* **[Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)** - Crucial for Week 2 & 3 tensor operations.
* **[Tiktokenizer](https://tiktokenizer.vercel.app/)** - A visual tool to understand how GPT sees text (BPE Tokenization).
## Getting Started & Workflow

We use a **Fork & Push** workflow. You will code in Colab, but you must save your work to your GitHub fork to submit it.

1. **Fork this repository** to your own GitHub account.
2. **Clone your fork** to your local machine (optional) or simply work via browser.
3. **Start Week 1:**
   * Open the Week 1 notebook link (provided in `weeks/week1.md`) in **Google Colab**.
   * `File` -> `Save a copy in Drive` to create your own editable version.
4. **Submission:**
   * When you finish the assignment in Colab: `File` -> `Save a copy in GitHub`.
   * Select your **Forked Repository** as the destination.
   * Add a commit message (e.g., "Completed Week 1 Assignment").

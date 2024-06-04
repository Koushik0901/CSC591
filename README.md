# CSC 591 - Scaling Large Language Models on GPUs

_A directed study course on how to scale out the training of large language models with hundreds of GPUs as required for GPT-3_

© 2024 Koushik Sivarama Krishnan & Sean Chester.

## Overview

In the past few years, the capacity of computer vision and natural language processing applications has exploded. The resulting "large language models" (LLMs) achieve a tremendous leap in performance on a wide range of machine learning tasks. A series of research breakthroughs led to the ML architecture, including work by Google to first introduce word embeddings [2] and then transition to transformer-based architectures like BERT and GPT [3]. 

These machine learning results have obviously received a lot of attention, but another profound enabling technology was required too. To train GPT-3 with a massively parallel Nvidia Volta GPU would still take 288 years [1]. In other words, without huge leaps in parallel and distributed computing, especially with the use of GPUs, LLMs would only be a theoretical and implausible idea.

This course will trace through the development in GPU and multi-GPU computing that lead up to the point that huge LLMs like GPT-3 could be trained.

## References

[1] Narayanan et al. (2021) "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." _SC '21: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_. https://doi.org/10.1145/3458817.3476209.

[2] Mikolov et al. (2013) "Distributed Representations of Words and Phrases and their Compositionality." _Advances in Neural Information Processing Systems 26 (NIPS 2013)_. https://papers.nips.cc/paper_files/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html

[3] Vaswani et al. (2017) "Attention Is All You Need." _31st Conference on Neural Information Processing Systems (NIPS 201)_. https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
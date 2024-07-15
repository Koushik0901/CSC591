# Training LLMs in CUDA

_Fundamentals of large language models (LLMs) and how to train them with GPUs_

## Key Objectives

 1. To be able to implement the ML architecture for training word embeddings for text corpora
 2. To be able to implement the ML architecture for training small-scale large language models
 3. To accelerate part of an LLM training pipeline in python/NLTK, pyTorch, and CUDA

## Key Deliverables

|Name  |Date        |Description|
|------|------------|-----------|
|Report|24 July 2024|A report describing a few implementations of training models to learn word embeddings and LLMs using python libraries and raw CUDA|

The report should be in .pdf format and not longer than six pages. It is recommended, but not required, to use [the standard format of the Association of Computing Machinery (ACM)](https://www.acm.org/publications/proceedings-template), preferably in double-column layout.
The focus of the report should be on describing how you worked through these milestones. 
The grade will be the percentage of milestones achieved.

In preparing the report, you should work through the following milestones:

 - [ ] Train word embeddings with existing python libraries like NLTK
 - [ ] Hand-code a model in python to train word embeddings and compare the quality to ones generated from NLTK
 - [ ] Train a small-scale LLM with existing python libraries like NLTK and pytorch
 - [ ] Hand-code part of the transformer architecture in python
 - [ ] Port the hand-coded python to CUDA to accelerate training times and/or increase the size of the training corpus

## Tools & Resources

The following tools and resources might help get the project off the ground faster:
  * [A Dummy's Guide to Word2Vec](https://medium.com/@manansuri/a-dummys-guide-to-word2vec-456444f3c673)
  * [Word2Vec from Scratch](https://medium.com/@enozeren/word2vec-from-scratch-with-python-1bba88d9f221)
  * [NLTK API Docs](https://www.nltk.org/api/nltk.html)
  * [PyTorch API Docs](https://pytorch.org/docs/stable/index.html)


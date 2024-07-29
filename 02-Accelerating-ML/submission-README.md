# Abstract
This report aims to discuss the training methodologies for Large Language Models and the optimization of such processes by training word embeddings and small-scale LLMs with Python and CUDA. By doing so, we seek to familiarize ourselves with the concepts of word embeddings \& Large Language Models and how to train them with CUDA and GPUs. We will begin by training the GloVe model\cite{pennington-etal-2014-glove} using glove-python and NLTK libraries before developing a custom Python model that emulates the GloVe model. We then compare the quality of embeddings from both these methods. We also continued our analysis by hand-coding and training a small-scale LLM using PyTorch based on BERT \cite{devlin2019bertpretrainingdeepbidirectional} architecture; we utilized CUDA through PyTorch to improve the training speed. These experiments show the enhancements of the training time and efficiency of these LLMs when CUDA and GPUs are employed. This further supports the reasons for incorporating CUDA in the LLM training processes. In addition to establishing the technical viability of employing CUDA for LLM training, this report has also outlined pertinent and practical difficulties encountered while constructing and training these LLMs.

# Run GloVe Training
`pip3 install -r requirements.txt`

`python glove_implementation.py`

# Run Bert Training
`pip3 install -r requirements.txt`

`python bert-from-scratch.py`

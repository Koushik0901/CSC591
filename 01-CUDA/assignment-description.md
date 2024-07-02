# CUDA GPU Programming Assignment — Assignment 2

_In this assignment you will implement and optimise a CUDA implementation of matrix multiplication, a core sub-routine for many machine learning applications._

## Objective

You should write a report that demonstrates the following competencies:

 1. To be able to describe solutions to problems using the SIMT model of parallelism popularised by Nvidia/CUDA
 2. To be able to implement basic CUDA programs that have applications to machine leaerning, such as matrix multiplication
 3. [optional] To understand how tensor cores differ from typical CUDA cores and how each of these differ from traditional CPU cores

In particular, the report should describe how to solve matrix multiplication within the CUDA/SIMT model and several iterations to optimise that solution for better performance and throughput.
  
## Submission

You should extend your github repository with your new code and share a link
to this and a commit hash corresponding to your submission. The repository
should contain a README with a description of the project, build instructions,
a license, C++/CUDA source code for each version of your matrix multiplication code (not necessarily separate files), any citations
to references that you have used. 
The report should be at most six pages that
contains algorithm descriptions and the results of performance experiments.

For an idea of how to plot performance experiments, see Figure 15 of the following manuscript: https://arxiv.org/pdf/2005.14469. Your trendlines will correspond to the different optimised versions of matrix multiplication that you have created.

This should be submitted via email or Github pull request no later than **23.59 UTC-7 
Wednesday 3 July 2024**.

## Report Structure

The structure of the report is not explicitly graded, but nonetheless you should follow a standard template for a scientific paper, including:

  * A title, author, and affiliation at the top of the first page
  * A copyright statement in the bottom left or footer of the first page
  * A short abstract of 1-3 paragraphs describing the purpose and results of the report
  * An introduction that describes in one page or less what problem you are trying to solve, briefly what are the key ideas you use to solve it, and briefly how you evaluate that
  * A background section in which you describe your first/baseline solution and how you designed it for CUDA
  * A methods section in which you describe the optimisations that you applied, preferably with a different subsection for each iteration of the design
  * An evaluation section in which you describe where your data comes from, what type of GPU you use, plots of experiment results and a short discussion or analysis of the results
  * A conclusion of 1-2 short paragraphs recapping the key points of the manuscript
  * A list of references

It is recommended, but not required, to follow [the ACM double-column conference paper layout](https://www.acm.org/publications/proceedings-template).

## Evaluation

Grades for this assignment will be based on four equal-weighted subjective criteria:
  
_Clarity of parallelisation strategy_  
It should be possible for a grad student in my research lab to understand with relative ease how you adapted matrix multiplication for CUDA  
  
_Clarity of optimisation strategies_  
It should be possible for a grad student in my research lab to understand with relative ease at least two optimisations that you have applied in order to improve performance  
  
_Performance results_  
There should be noteworthy (e.g., 10x?) faster performance for your most optimised version relative to your initial baseline solution
  
_Scientific reproducibility_  
It should be possible for a grad student in my research lab to recreate the scientific plots with less than an hour's effort.



## Academic Integrity and Sources

You are welcome to utilise any sources that you find on the internet so long
as they are cited clearly. You are responsible for establishing the accuracy
of any such materials. Copying directly or using generative AI will undermine the efficacy of
this training module and is very strongly discouraged.

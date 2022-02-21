# cross-dpc-episdet

This repository contains several implementations of exhaustive epistasis detection for third-order interaction searches that support single-objective evaluation with Bayesian K2 score scoring function. These implementations target CPUs and GPUs from Intel, AMD and NVIDIA, and integrate different optimization techniques that aim at maximizing the performance of the application in current computing devices. The CPU implementations are parallelized by using OpenMP, while GPU kernels are deployed with the DPC++ programming model.

## What is Epistasis Detection?

Epistasis detection is a computationally complex bioinformatics application with significant societal impact. It is used in the search of new correlations between genetic markers, such as single-nucleotide polymorphisms (SNPs), and phenotype (e.g. a particular disease state).
Finding new associations between genotype and phenotype can contribute to improved preventive care, personalized treatments and to the development of better drugs for more conditions.

## Description

The repository is structured as follows:

* `CPU`: Contains the CPU implementations.
* `GPU`: Contains the GPU implementations.

## Setup

### Requirements

* DPC++ (version 1.2 or more recent)
* OpenMP
* g++ (only tested with 9.3)

### Compilation



## Usage example


## In papers and reports, please refer to this tool as follows

Marques D., Campos R., Santander-Jiménez S., Matveev, Z., Sousa L., Ilic A. Unlocking Personalized Healthcare on Modern CPUs/GPUs: Three-way Gene Interaction Study. In: 36th IEEE International Parallel & Distributed Processing Symposium (IPDPS 2022).

# cross-dpc-episdet

<p>
  <a href="https://doi.org/10.1109/IPDPS53621.2022.00023" alt="Publication">
    <img src="https://img.shields.io/badge/DOI-10.1109/IPDPS53621.2022.00023-blue.svg"/></a>
    
</p>

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
* NVIDIA CUDA for deployment in NVIDIA GPUs (only tested with CUDA 11.4)
* AMD ROCm for deployment in AMD GPUs (only tested with ROCm 4.2)
* Intel Level-Zero or OpenCL for deployment in Intel GPUs (only tested with Intel Level-Zero backend)

### Compilation and Usage

Check the README.md in CPU and GPU folders.

## In papers and reports, please refer to this tool as follows

Marques D., Campos R., Santander-Jiménez S., Matveev, Z., Sousa L., Ilic A. Unlocking Personalized Healthcare on Modern CPUs/GPUs: Three-way Gene Interaction Study. In: 36th IEEE International Parallel & Distributed Processing Symposium (IPDPS 2022).

BibTeX:

    @INPROCEEDINGS{9820625,
    author={Marques, Diogo and Campos, Rafael and Santander-Jiménez, Sergio and Matveev, Zakhar and Sousa, Leonel and Ilic, Aleksandar},
    booktitle={2022 IEEE International Parallel and Distributed Processing Symposium (IPDPS)}, 
    title={Unlocking Personalized Healthcare on Modern CPUs/GPUs: Three-way Gene Interaction Study}, 
    year={2022},
    pages={146-156},
    doi={10.1109/IPDPS53621.2022.00023}
    }


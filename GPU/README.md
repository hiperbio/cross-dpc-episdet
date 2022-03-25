# GPU Implementations

This folder contains the GPU implementations considered in the article. Below you can find a summary of each version:

* `V1`: Baseline implementation with binarized input dataset.
* `V2`: The phenotype is removed from the computation by separating the dataset in controls and cases. The third genotype is inferred from the other two.
* `V3`: Integrates data set transposing to improve memory accesses.
* `V4`: Integrates data set tiling techniques to improve memory accesses.

## Compilation

The CPU base frequency is used to obtain an accurate time reading with the TSC counter. For that, it is necessary to define the FREQ macro in the code (e.g., `#define FREQ 3500000000` for a base frequency of 3.5 GHz).
To build all the versions:
`make all`
To build a specific version:
`make v<1-4>`

## Usage

For versions `V1`, `V2`, `V3`:
`./v<1-3> <num_samples> <num_snps>`
For version `V4`:
`./v4 <num_samples> <num_snps> <block_size>`

# GPU Implementations

This folder contains the GPU implementations considered in the article. Below you can find a summary of each version:

* `V1`: Baseline implementation with binarized input dataset.
* `V2`: The phenotype is removed from the computation by separating the dataset in controls and cases. The third genotype is inferred from the other two.
* `V3`: Integrates data set transposing to improve memory accesses.
* `V4`: Integrates data set tiling techniques to improve memory accesses.

## Compilation

The CPU base frequency is used to obtain an accurate time reading with the TSC counter. For that, it is necessary to define the FREQ macro in the code (e.g., `#define FREQ 3500000000` for a base frequency of 3.5 GHz).

To build the application:
`make <version>`

Where `<version>` is one of `v1`, `v2`, `v3`, `v4` or `all`.

When targetting Intel GPUs, no further options are necessary. For the NVIDIA and AMD GPUs tested, it is necessary to add some compiler options like follows:
`make <version> gpu=<target>`

Where `<target>` is one of `nvidia` (for NVIDIA Titan V, Titan RTX and A100), `nvidia_titan_xp`, `amd_vega_20`, `amd_instinct_mi100`, or `amd_navi_21`.

## Usage

For versions `V1`, `V2`, `V3`:
`./<version> <num_samples> <num_snps>`

For version `V4`:
`./v4 <num_samples> <num_snps> <block_size>`

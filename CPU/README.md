# CPU Implementations

This folder contains the CPU implementations considered in the article. Below you can find a summary of each version:

* `V1`: Baseline implementation with binarized input dataset.
* `V2`: The phenotype is removed from the computation by separating the dataset in controls and cases. The third genotype is inferred from the other two.
* `V3`: Integrates cache blocking techniques.
* `V4_AVX`: Vectorized with AVX vector intrisics for processors that do not support vectorized population count instructions.
* `V4_AVX512`: Vectorized with AVX512 vector intrisics for processors that do not support vectorized population count instructions.
* `V4_POPC_AVX512`: Vectorized with AVX512 vector intrisics for processors that support vectorized population count instructions (IceLake-SP).

## Compilation

To build the application:

`make <version> freq=<frequency_in_hz> threads=<num_threads>` 

where `<version>` is one of `v1`, `v2`, `v3`, `v4_avx`, `v4_avx512`, `v4_popc_avx512` or `all`. The parameter `threads` is the number of threads to launch (by default threads=1). The input `freq` is the nominal frequency of the CPU in Hz and it is used to calculate the time from the cycles measured with the TSC counter.

## Usage

For versions `V1`, `V2`:
`./<version> <num_samples> <num_snps>`

For versions `V3`, `V4_AVX`, `V4_AVX512`, `V4_POPC_AVX512`:
`./<version> <num_samples> <num_snps> <block_samples> <block_snps>` 

where <num_samples> are the number of patients in the dataset, <num_snps> the number SNPs in the dataset, while <block_samples> and <block_snps> are the blocking factors for the samples and SNPs used in the versions with cache blocking techniques.

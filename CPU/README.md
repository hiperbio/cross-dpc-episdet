# CPU Implementations

This folder contains the CPU implementations considered in the article. Below you can find a summary of each version:

* `V1`: Baseline implementation with binarized input dataset.
* `V2`: The phenotype is removed from the computation by separating the dataset in controls and cases. The third genotype is inferred from the other two.
* `V3`: Integrates cache blocking techniques.
* `V4_AVX`: Vectorized with vector intrisics for processors that do not support vectorized population count instructions.
* `V4_AVX512`: Vectorized with vector intrisics for processors that support vectorized population count instructions (IceLake-SP).

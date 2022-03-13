# CPU Implementations

This folder contains the CPU implementations used in the evaluation with the Cache-Aware Roofline Model. Below you can find a summary of each version:

* `V1`: Baseline implementation with binarized input dataset.
* `V2`: The phenotype is removed from the computation by separating the dataset in controls and cases. The third genotype is inferred from the other two.
* `V3`: Integrates cache blocking techniques.
* `V4`: Vectorized with vector intrisics.

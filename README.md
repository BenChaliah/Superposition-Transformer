# Superposition in Transformers: A Novel Way of Building Mixture of Experts

> __If you find this work interesting, please consider starring this repository!__

Paper available on arXiv: [Superposition in Transformers: A Novel Way of Building Mixture of Experts](https://arxiv.org/abs/2501.00530)

## Repository Contents

- `Superposition_in_Transformers.pdf` - The full research paper
- `1D-Alpha_Variant_LayerBias_LinearConv.ipynb` - Implementation of the 1D-alpha model with layer bias and linear convolution
- `2D-Alpha_Variant_LayerBias_ResLinearAdapter-Conv.ipynb` - Implementation of the 2D-alpha model with layer bias and residual linear adapter convolution
- `Benchmarks_1DAlpha_LayerBias_ConvLinear.ipynb` - Performance benchmarks and comparisons for the 1D-alpha model

### Abstract

Catastrophic forgetting remains a major challenge when adapting large language models (LLMs) to new tasks or domains. We introduce *Superposition in Transformers*, a novel architecture that leverages autoencoders to superimpose the hidden representations of a base model and a fine-tuned model within a shared parameter space. Using B-spline-based blending coefficients and autoencoders that adaptively reconstruct hidden states based on the input data distribution, our method effectively mitigates catastrophic forgetting and enables a new paradigm of "in-model" superposition.

### Notebooks Overview

1. **1D-Alpha_Variant_LayerBias_LinearConv.ipynb**  
    Demonstrates the "1D-alpha model" using scalar α values for each layer.
    - B-spline-based α blending implementation
    - Autoencoder usage for reconstructing base/fine-tuned hidden states
    - Perplexity and accuracy metrics for English-French adaptation

2. **2D-Alpha_Variant_LayerBias_ResLinearAdapter-Conv.ipynb**  
    Explores the "2D-alpha model" with vector-based α per dimension.
    - Local (convolutional) and global (adapter) autoencoder pathways
    - Polysemantic neuron analysis and multi-task representation
    - t-SNE visualizations of hidden states

3. **Benchmarks_1DAlpha_LayerBias_ConvLinear.ipynb**  
    - Performance comparisons against baselines
    - Perplexity and Jensen-Shannon divergence analysis
    - Direct comparisons with linear interpolation methods

## Key Features

- **Autoencoder-Based Superposition**: Hidden states from base and fine-tuned models are combined and reconstructed by autoencoders, preserving domain-specific knowledge.
- **B-Spline Blending**: Smooth transitions between base and fine-tuned states using learned blending coefficients.
- **Parameter Efficiency**: Only trains small auxiliary components while keeping main model weights frozen.
- **Polysemantic Neurons**: Demonstrates emergence of neurons that handle multiple tasks/domains.

## Environment Setup

```bash
pip install torch transformers scikit-learn matplotlib
```

## Results

Our experiments show:
- Lower perplexity vs linear interpolation and task arithmetic models
- Preservation of both English and French capabilities
- Increased polysemantic neuron count
- Successful dynamic language model switching during inference

## Citation

```bibtex
@misc{benchaliah2023superposition,
   title={Superposition in Transformers: A Novel Way of Building Mixture of Experts},
   author={Ben Chaliah, Ayoub and Dellagi, Hela},
   year={2024},
   eprint={2501.00530},
   archivePrefix={arXiv},
   primaryClass={cs.LG},
   url={https://arxiv.org/abs/2501.00530},
   howpublished={\url{https://github.com/BenChaliah/Superposition-Transformer}},
}
```

## Contributing

We welcome:
- Issues via GitHub Issue Tracker
- Pull requests with improvements
- Feedback and suggestions

## Authors

- Ayoub Ben Chaliah - [ayoub1benchaliah@gmail.com](mailto:ayoub1benchaliah@gmail.com)
- Hela Dellagi - [hela.dellagi@outlook.com](mailto:hela.dellagi@outlook.com)

---
### License: MIT
---

**Enjoy exploring Superposition in Transformers!**

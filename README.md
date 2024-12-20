# Superposition-Transformer


## Overview

This repository contains the implementation of our method for creating Mixture of Experts (MoE) models where multiple Language Models co-exist in the same parameter space. Our approach uses autoencoders and B-spline blending to enable efficient model merging while mitigating catastrophic forgetting.

## Important Disclaimers

- This implementation is a research prototype and may not be suitable for production use
- The models' behavior and edge cases are not fully understood - carefully evaluate for your use case
- Performance may vary depending on the specific models being merged and domains involved

## Repository Contents

The implementation is provided in two Jupyter notebooks:

### 1. `1D_alpha_model.ipynb`
- Implementation of the scalar alpha value architecture
- Complete training pipeline
- Evaluation metrics and validation
- Visualization code for:
  - t-SNE plots
  - Hidden state trajectories
  - Model performance analysis

### 2. `2D_alpha_model.ipynb`
- Implementation of the vector-based alpha architecture
- Training and evaluation pipeline
- Analysis tools for neuron polysemanticity
- Visualization code for:
  - Neuron diversity plots
  - Activation patterns
  - Enhanced t-SNE visualizations

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- Transformers
- NumPy
- Scikit-learn
- Matplotlib
- Pandas

## Experimental Results

Our implementation demonstrates:
- Effective merging of English and French language models
- Perplexity metrics comparable to specialized models
- Successful hidden state reconstruction
- Increased proportion of polysemantic neurons

For detailed results and analysis, please refer to the paper.

## Future Work

We are considering:
- Extending the approach to handle more than two models
- Investigating bottleneck hyperparameters and their impact on model performance
- Exploring applications in chain-of-thought reasoning

## Citation

Please use the following bibtex entry:
```bibtex
@article{benchaliah2024superposition,
  title={Superposition and Autoencoders, a Novel Way of Building Mixture of Experts},
  author={Ben Chaliah, Ayoub and Dellagi, Hela},
  year={2024}
}
```

## Contact

For questions and feedback:
- Ayoub Ben Chaliah - ayoub1benchaliah@gmail.com
- Hela Dellagi - hela.dellagi@outlook.com

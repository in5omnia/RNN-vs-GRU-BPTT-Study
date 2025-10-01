# NLU+ Coursework 1: Recurrent Neural Networks for Language Modeling

Implementation of Recurrent Neural Networks (RNNs) and Gated Recurrent Units (GRUs) for language modeling and syntactic agreement prediction tasks, built from scratch using NumPy.
This project was done with [@ankurc561](https://github.com/ankurc561).

## Project Overview

This project implements fundamental components of recurrent neural networks without relying on high-level deep learning frameworks, focusing on:

1. **Language Modeling**: Predicting the next word in a sequence using RNNs
2. **Agreement Prediction**: Testing RNN capabilities on syntactic number agreement tasks
3. **Long-Range Dependencies**: Comparing RNN and GRU performance with varying backpropagation through time (BPTT) steps

## Implementation Details

### What We Implemented

#### 1. RNN from Scratch (`rnn.py`)
Built a vanilla RNN with three main components:
- Forward pass that processes sequences word-by-word
- Truncated backpropagation (1-step) for basic gradient computation
- Extended backpropagation through time (BPTT) with configurable lookback
- Variants adapted for binary classification (agreement prediction)

#### 2. GRU from Scratch (`gru.py`)
Implemented a Gated Recurrent Unit with:
- Forward pass using reset and update gates
- Integration with provided backpropagation framework
- Binary classification variants for agreement tasks

#### 3. Training Infrastructure (`runner.py`)
Developed complete training loops with:
- Cross-entropy loss computation for language modeling
- Binary classification loss and accuracy metrics
- Support for multiple training modes (language modeling, agreement prediction)
- Hyperparameter scheduling (learning rate annealing)

### What We Analysed (RNN vs GRU Comparison)
1. Vanishing Gradient Analysis
2. General Performance Validation vs. Lookback (number of BPTT steps):
   - Average Time To Train
   - Run Loss
   - Accuracy
3. Effect of Hyperparameters (reporting Run Loss):
   - Training Size (also reporting Count vs Distance between subj and verb.)
   - Number of Epochs
   - Number of Hidden Dimensions
   - Batch Size (also reporting Average Time To Train vs Lookback)
   - Learning Rate

## Key Experimental Findings

### Language Modeling Performance

**Best Configuration** (1,000 sentences, 30 epochs):
- Hidden units: 50
- Learning rate: 0.8
- BPTT steps: 3
- Batch size: 5
- Anneal: 2
- Epochs: 30
- **Best loss**: 4.530544
- **Perplexity**: 92.809

**Key Insights**:
- **GRU vs RNN:** GRU consistently outperforms vanilla RNN in loss and accuracy, especially for longer BPTT steps, due to its gating mechanisms mitigating vanishing gradients.
- **Training Trade-offs:** GRU requires more computation per epoch; small batch sizes and higher learning rates improve generalization and convergence stability.
- **Data & Model Capacity:** Moderate hidden dimensions (50) balance model expressiveness and overfitting; excessive epochs or hidden units may degrade performance due to limited data.
- **BPTT Effects:** RNN stagnates at higher BPTT steps due to vanishing gradients, whereas GRU can leverage longer dependencies when properly tuned.

## Repository Structure

```
.
├── code/
│   ├── rnn.py                    # RNN implementation
│   ├── gru.py                    # GRU implementation  
│   ├── gru_abstract.py           # GRU backpropagation framework
│   ├── model.py                  # Abstract base class for models
│   ├── runner.py                 # Training loops and loss functions
│   ├── rnnmath.py                # Helper mathematical functions
│   ├── utils.py                  # Data processing utilities
│   ├── test.py                   # Unit tests
│   └── requirements.txt          # Python dependencies
└── report.pdf                    # Written analysis and experimental results
```


## Notes
- This repository is an **academic coursework submission**.  
- Code and report are included for archival/reference purposes.  
- It is **not intended for direct reuse** without adaptation.  
- All implementations follow the coursework specification.  
- The dataset used was provided as a zip with the coursework specification.

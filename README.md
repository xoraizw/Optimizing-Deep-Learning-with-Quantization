# Quantization: Exploring Trade-offs Between Precision and Performance

Quantization is a model compression technique that reduces the precision of weights and activations, enabling significant memory and computational efficiency. This repository focuses on evaluating and comparing two key quantization strategies:

1. **Post-Training Quantization (PTQ)**  
2. **Quantization-Aware Training (QAT)**  

These experiments are essential for optimizing deep learning models for deployment on edge devices or specialized hardware.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Project Highlights](#project-highlights)
- [Getting Started](#getting-started)
- [Learnings](#learnings)
- [Results and Insights](#results-and-insights)
- [References](#references)

## Overview

### Objective
- Investigate the impact of reducing precision to bit-widths such as `fp16`, `bf16`, `int8`, and `int4`.
- Analyze accuracy, performance, and memory efficiency trade-offs.
- Explore how QAT can recover performance lost due to PTQ.

### Tasks
1. **PTQ & QAT**  
   - Quantize a pre-trained VGG-11 model fine-tuned on CIFAR-100.
   - Compare model performance across different bit-widths.
2. **Scaling Law Analysis**  
   - Examine accuracy trends as bit precision decreases.
3. **FP16 vs BF16 Analysis**  
   - Study the differences in memory footprint and numerical stability.

## Technologies Used
- **Frameworks & Libraries**: PyTorch, Torchvision, TorchAo, Matplotlib
- **Dataset**: CIFAR-100
- **Hardware**: CUDA-enabled GPUs for accelerated computations
- **Tools**: Jupyter Notebook, tqdm, pandas

## Project Highlights

### Quantization Techniques
- **Post-Training Quantization (PTQ)**: Directly reduces precision after training, offering simplicity but often at the cost of accuracy.
- **Quantization-Aware Training (QAT)**: Incorporates quantization effects during training to adapt the model and recover performance.

### Bit-width Evaluations
- Investigated `fp16`, `bf16`, `int8`, and `int4` representations.
- Plotted accuracy vs. bit-width to identify trade-offs.

### Advanced Insights
- Observed the memory vs. accuracy trade-offs between `fp16` and `bf16`.
- Evaluated the number of epochs required for QAT to match PTQ accuracy.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.10+
- CUDA 11.0+ (optional but recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/quantization-exploration.git
   cd quantization-exploration
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the CIFAR-100 dataset automatically during runtime.

### Usage
- To run the experiments, execute:
  ```bash
  python main.py
  ```

## Learnings

- **Quantization Impact**: Lower bit-widths such as `int4` significantly reduce memory usage but introduce accuracy drops.
- **QAT Benefits**: QAT effectively recovers accuracy, especially for lower bit-widths (`int4` and `int8`), albeit at a computational cost.
- **FP16 vs BF16**: While `bf16` provides more stable training due to its dynamic range, `fp16` achieved slightly better accuracy and required less runtime.

## Results and Insights

### Accuracy vs Bit-width
- PTQ yielded:
  - `fp16`: 70.8%
  - `bf16`: 70.73%
  - `int8`: 70.78%
  - `int4`: 69.05%

- QAT improved `int4` to 69.27% and `int8` to 70.22%, nearing full precision accuracy.

### Memory Footprints
- `fp16`: 18.69 MB  
- `bf16`: 18.70 MB  

Both had comparable memory footprints, with `bf16` exhibiting greater numerical stability.

## References
- [TorchAo Quantization Library](https://pytorch.org/docs/stable/quantization.html)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

---

Would you like to include any specific visualizations or additional details?

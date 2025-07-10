# Structured Pruning + Post-Training Quantization on Fashion-MNIST

## Project Title
**“Slash & Shrink: 50 % Pruned, INT8 Quantized CNN”**

## Description
This mini-project shows how to compress a convolutional network via:

1. **Structured weight pruning (filter pruning)** to 50 % sparsity  
2. **Dynamic-range INT8 quantization** (TensorFlow Lite)  

We train a small CNN for two epochs on Fashion-MNIST, then apply pruning and quantization.  
The script prints test accuracy and disk size for each stage.

## Dataset
- **Fashion-MNIST** (70 000 grayscale images, 10 classes)  

## Results
```bash
=== Compression Results ===
            Model  Accuracy  Size_MB
   Baseline FP32    0.9112     1.276
     Pruned FP32    0.9043     0.879
 Pruned + INT8      0.9043     0.332
Run for 2.15 minutes on 4070 GPU

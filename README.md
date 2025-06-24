# AEIOU Vowel Classification
This project implements a 1D Convolutional Neural Network (CNN) in PyTorch to classify English vowel sounds — A, E, I, O, U — based on raw signal input. It is designed for simple end-to-end training and evaluation using preprocessed `.txt` data.

**Note**: This project was developed as part of a university coursework assignment for deep learning and signal processing.  
It focuses on classifying vowel sounds (A, E, I, O, U) using a 1D CNN model implemented in PyTorch.
> Course: Deep Learning Theory & Exercise <br>
> Instructor: Prof. AHN <br>
> Submitted by: [SOHN] <br>
> Semester: Spring 2025 <br>
---

# Prerequisites
some package required:
-Python 3.7+
-PyTorch
-NumPy
-Matplotlib

# Project Structure
```
python/
├── AEIOU_classification/
│   ├── AEIOU_classification.ipynb     # Main notebook (training & evaluation)
│   └── checkpoints/                   # Saved best models by accuracy
├── DL/
│   └── aeiou_data/
│       ├── 75Raw_a.txt
│       ├── 75Raw_e.txt
│       ├── 75Raw_i.txt
│       ├── 75Raw_o.txt
│       └── 75Raw_u.txt
```

# Model Overview
The model is a deep 1D CNN that:
- Accepts 1-channel input of shape `(batch, 1, 20000)`
- Applies 7 convolutional layers + max pooling
- Ends with a fully connected layer for classification into 5 vowel classes

You can find the full model in the notebook:
```python
class CNN(nn.Module):
   ...
```

# Data Preparation
Each txt file contains 75 samples (20000 values each) for a single vowel.
-First 50 samples per vowel → used for training
-Last 25 samples per vowel → used for testing
The data is reshaped and converted to torch.FloatTensor for model input.

# Output
-Training & test loss/accuracy are plotted at the end
-Model with highest test accuracy is saved during training

# Checkpointing
Best model is automatically saved to: checkpoints/best/best_model_epochXX_accYY.YY.pt
Additional metadata is stored in best_info.txt.

# DILATE-Loss-TensorFlow
This repository provides a TensorFlow implementation of the DILATE loss, a differentiable loss function designed for time series alignment tasks. DILATE combines shape alignment using Soft Dynamic Time Warping (SoftDTW) and a temporal regularization term to ensure smooth and interpretable alignment paths. The implementation is fully differentiable and can be integrated into TensorFlow-based models for supervised learning of time series alignment. Example scripts and tests are included for reproducibility and ease of use.


### Features
- **SoftDTW (Soft Dynamic Time Warping):** A differentiable approach to compute time series shape alignment.
- **Temporal Regularization:** A penalty for alignment paths to ensure temporal smoothness.


### Installation
To install the necessary dependencies, clone this repository and install the required packages:
```bash
   git clone https://github.com/shk36/DILATE-Loss-TensorFlow.git
   cd DILATE-Loss-TensorFlow
```

### Usage
An example usage of the DILATE loss function:
```python
  import tensorflow as tf
  from dilate_loss.dilate_loss_tf import dilate_loss
  
  # Example inputs
  batch_size = 2
  N_output = 5
  outputs = tf.random.uniform((batch_size, N_output, 1), dtype=tf.float32)
  targets = tf.random.uniform((batch_size, N_output, 1), dtype=tf.float32)
  alpha = 0.5
  gamma = 1.0
  
  # Compute DILATE loss
  loss, loss_shape, loss_temporal = dilate_loss(outputs, targets, alpha, gamma)
  
  print("Loss:", loss.numpy())
  print("Shape Loss:", loss_shape.numpy())
  print("Temporal Loss:", loss_temporal.numpy())
```

### References
- Le Guen, V., & Thome, N. "[Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models](https://arxiv.org/abs/1909.09020)" (2019).

# STLNet: Shallow Texture-Enhanced Lightweight Network for Face Forgery Detection

This repository provides the official PyTorch implementation of STLNet,
a lightweight face forgery detection network designed to capture
fine-grained shallow texture anomalies while maintaining efficient
inference.

STLNet consists of three main components:

- **Shallow Feature Extraction Module (SFE):** extracts multi-scale
  shallow texture representations using Lightweight Shallow Texture
  Blocks.
- **Texture Enhancement Module (TEM):** strengthens forgery-sensitive
  regional texture responses through TRU and APU.
- **Lightweight Classifier Module (LCM):** performs efficient binary
  classification using a re-parameterizable classifier and residual
  adapter.

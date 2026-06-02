# CFracFormer

Lightweight PyTorch implementation of **Closed-loop Fractal-domain Transformer Model for SAR Ship Classification**.

CFracFormer extends FracFormer with a closed-loop fractal reconstruction branch. The model contains four main parts:

- ViT-style image embedding
- SPHS/SIFT fractal feature extraction
- mask attention and fractal reconstruction regularization
- Transformer feature mixing and classification head

The paper evaluates the model on OpenSARShip2.0 SAR ship recognition. The default code is configured for 3-class dual-polarization input with shape `[B, 2, 64, 64]`.

## Structure

```text
CFracFormer-GitHub/
  models/
    cfracformer.py
  weights/
    reconstruction_model_mse.pth
    reconstruction_model_ssim.pth
  requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
```

## Notes

- `reconstruction_model_mse.pth` and `reconstruction_model_ssim.pth` are reconstruction checkpoints kept for reference.
- Dataset files are not included. Prepare OpenSARShip samples as tensors shaped `[2, 64, 64]`.


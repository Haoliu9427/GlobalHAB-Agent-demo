# Optional adaptive visual assets

The core GlobalHAB-Agent package starts without PyTorch and without any external downloads.
Deep visual branches are activated **only** when both a feature encoder and a project-trained
lightweight head are available.

Supported branches:

- `DINOv2`: local Hugging Face-compatible directory at `vision_models/dinov2_local/`, or set `GLOBALHAB_DINOV2_MODEL_DIR`.
- `ConvNeXt-Tiny`: local checkpoint `vision_models/convnext_tiny.pth`, or set `GLOBALHAB_CONVNEXT_CHECKPOINT`.
- `EfficientNet-B0`: local checkpoint `vision_models/efficientnet_b0.pth`, or set `GLOBALHAB_EFFICIENTNET_CHECKPOINT`.

For development only, set `GLOBALHAB_ALLOW_MODEL_DOWNLOADS=1` to allow torchvision/Hugging Face
pretrained weights to be downloaded when the optional dependencies are installed. Production
releases should prefer frozen local assets.

Each encoder also needs a matching NumPy head bundle under `vision_models/heads/`. Do not copy an
ImageNet classifier and call it a HAB model: train/calibrate the head on project-labelled field
images with site/time holdouts.

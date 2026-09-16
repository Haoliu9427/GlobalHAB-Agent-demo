# Visual model assets

This directory stores optional local visual assets for the field-image workspace.

- `efficientnet_b0.pth`: optional local EfficientNet-B0 checkpoint. If absent, the app first tries the public Torchvision pretrained weights; in a fully offline non-strict deployment it can use the explicitly labelled deterministic prototype-encoder initialisation.
- `convnext_tiny.pth`: optional local ConvNeXt-Tiny checkpoint with the same resolution order.
- `dinov2_local/`: optional local Hugging Face-compatible DINOv2-small directory. DINOv2 has no random-weight fallback; it participates only when a real local/cache/downloaded model is available.
- `heads/*.npz`: optional project-trained fusion heads. When missing, the built-in visual-phenomenon prototype head is used instead.

Environment variables:

```text
GLOBALHAB_EFFICIENTNET_CHECKPOINT
GLOBALHAB_CONVNEXT_CHECKPOINT
GLOBALHAB_DINOV2_MODEL_DIR
GLOBALHAB_ALLOW_MODEL_DOWNLOADS=1
GLOBALHAB_STRICT_PRETRAINED=0
```

The built-in prototype head is a screening mechanism, not a validated HAB species/toxin classifier. A real project-trained head should be produced from labelled field photographs with independent site/year validation before reporting scientific classification performance.

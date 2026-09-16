# Lightweight visual head format

Each branch uses a safe NumPy `.npz` file named `dinov2.npz`, `convnext.npz` or `efficientnet.npz`.
The bundle contains:

- `coef`: shape `[n_classes, embedding_dim + metadata_dim]`
- `intercept`: `[n_classes]`
- `mean`, `scale`: feature standardisation vectors
- `classes`: project visual class names
- `ood_threshold`: optional scalar RMS z-distance threshold

The runtime app will not activate a deep branch when its trained head is absent. This prevents
untrained foundation features from being presented as validated HAB predictions.

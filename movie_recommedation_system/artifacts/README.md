# artifacts/

This folder contains **trained models and configuration files** that the Streamlit app loads at runtime.

## Files
- `svd_model.pkl`
  - The trained **collaborative filtering** model (Surprise SVD).
  - Used in `app.py` to predict a user’s rating for a movie.
- `svd_metadata.pkl`
  - A small dictionary of **training metadata** (example: params, CV metrics, dataset sizes).
  - Useful for reporting/debugging; not required for inference.
- `hybrid_config.pkl`
  - Optional configuration describing hybrid defaults (example: default α).
  - Not required by the current app logic.
- `explanation_config.pkl`
  - Optional configuration describing explanation/version info.
  - Not required by the current app logic.

## Subfolders
- `content/`
  - Content-based artifacts (embeddings + metadata) used for “More Like This”.


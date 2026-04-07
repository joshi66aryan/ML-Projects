# data/processed/

This folder contains **processed/enriched datasets** produced by the notebooks and consumed by the Streamlit app.

## Files
- `enriched_movies.csv`
  - Main movie table used by the app (movieId, title, genres, overview, poster path, etc.).
  - Joined/enriched from MovieLens + TMDB metadata.
- `enriched_movies_sample.csv`
  - Smaller sample version of the enriched dataset.
  - Useful for quick iteration/demos.

Note: Large model artifacts like embeddings are stored under `artifacts/content/`.


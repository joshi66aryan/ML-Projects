# Smart Movie Recommender (Hybrid: Content + Collaborative)

## Overview
This project is a **hybrid movie recommender**:
- **Collaborative filtering (SVD)** learns from user–movie ratings (MovieLens) to predict what a user will like.
- **Content-based similarity (embeddings + cosine similarity)** finds movies that are semantically similar to a selected “seed” movie (plot/keywords/genres text).
- A **hybrid score** blends both signals so recommendations can be personalized *and* “more like this”.

The Streamlit app (`app.py`) loads **pre-trained artifacts** from `artifacts/` and precomputed data from `data/`.

## Architecture
High-level components:
- **Streamlit UI (`app.py`)**
  - Lets you pick a userId (collaborative focus) or a seed movie (content boost).
  - Displays recommendations, posters, and short explanations.
- **Artifacts (`artifacts/`)**
  - `svd_model.pkl` → trained collaborative filtering model.
  - `artifacts/content/movie_embeddings.npy` → semantic embeddings used for content similarity.
- **Data (`data/`)**
  - MovieLens ratings/movies data (inputs to training).
  - Enriched movie metadata used by the app (titles, genres, overview, poster path, etc.).
- **Notebooks (`movie_recommender/`)**
  - Training/experimentation pipeline that produces the artifacts and processed files.

Recommendation flow (simplified):
```text
User action
  ├─ For This User → SVD predicts ratings for unrated movies
  └─ More Like This → cosine_similarity(seed_embedding, all_embeddings)

Hybrid score = α * content_score + (1 - α) * collaborative_score
Top-N movies returned → rendered in Streamlit UI
```

## What “Content-Based” and “Collaborative (SVD)” Mean Here

### Collaborative filtering (SVD)
**Idea:** If two users rate similar movies similarly, they may like similar unseen movies.

**How it’s used in this project:**
- Training data: `data/ml-latest-small/ratings.csv` (userId, movieId, rating).
- Training notebook: `movie_recommender/03_Collaborative_FIiltering_Engine.ipynb`.
- Output artifact: `artifacts/svd_model.pkl` (Surprise SVD model).
- In the app: for a userId and each candidate movieId, the SVD model predicts a rating:
  - `pred = svd_model.predict(user_id, movie_id)`
  - Higher predicted ratings increase the collaborative part of the hybrid score.

What SVD “produces” that helps hybrid search:
- A per-user preference signal (predicted rating) even if a movie is not textually similar to a seed.
- A fallback when content similarity is weak or when no seed is provided.

### Content-based similarity (embeddings)
**Idea:** Represent each movie as a numeric vector so “similar meanings” are close together.

**How it’s used in this project:**
- Input text: the enriched metadata in `data/processed/enriched_movies.csv` (ex: overview/genres combined as `content_text`).
- Embedding notebook: `movie_recommender/02_Content_Based_Engine.ipynb` (uses Sentence-Transformers).
- Output artifact: `artifacts/content/movie_embeddings.npy` (one vector per movie).
- In the app: when you pick a seed movie, the app compares its embedding to all other movie embeddings using cosine similarity:
  - `sim = cosine_similarity(seed_embedding, candidate_embedding)`
  - Higher similarity increases the content part of the hybrid score.

## How It Works
1. **Load data & artifacts**
   - Reads `data/processed/enriched_movies.csv` for movie metadata.
   - Loads embeddings from `artifacts/content/movie_embeddings.npy`.
   - Loads the trained SVD model from `artifacts/svd_model.pkl`.
2. **Choose mode**
   - **For This User:** recommends based mostly on SVD predictions for a given `userId`.
   - **More Like This Movie:** selects a seed title and computes cosine similarity to boost content relevance.
3. **Score all candidate movies**
   - Skip movies already rated by the user.
   - Compute:
     - `collab_norm` from SVD predicted rating (normalized to 0–1)
     - `content_score` from cosine similarity (0–1-ish)
   - Blend into a hybrid score using `α` (content weight).
4. **Sort & display**
   - Sort by hybrid score and show top-N cards with poster + short explanation.

## Example Scenarios (What Happens in the App)

### 1) User provides **only a userId** (Collaborative-focused mode)
Use case: “Recommend movies tailored for user 42.”

What happens:
- The app finds which `movieId`s that user has already rated and skips them.
- For each remaining movie:
  - SVD predicts how much the user would rate it (`pred.est`).
  - Content similarity is not used because there is no seed movie.
- The top-N results tend to reflect the user’s learned tastes from ratings patterns.

### 2) User provides **only a seed movie** (Content-focused mode / “More Like This”)
Use case: “More like *Harry Potter and the Goblet of Fire (2005)*.”

What happens:
- The app takes the selected seed title and finds its row in `enriched_movies`.
- It grabs the seed’s embedding vector from `movie_embeddings.npy`.
- For each candidate movie:
  - Compute cosine similarity to the seed (content score).
  - Still compute SVD predicted rating for the chosen userId (because the app needs a userId for the hybrid blend),
    but in “More Like This” mode the content part is intentionally weighted very heavily.
- The top-N results are primarily “semantically similar” movies (same franchise/setting/themes).

### 3) User provides **both userId and seed movie** (Hybrid)
Use case: “More like *Lord of the Rings* **for user 42**.”

What happens (best of both worlds):
- Content similarity pulls the list toward movies similar to the seed.
- SVD re-ranks within those candidates toward movies the user is more likely to enjoy.

In other words:
- Content answers: “What is similar to this movie?”
- SVD answers: “What will this user like?”
- Hybrid answers: “What is similar **and** likely to be liked by this user?”

## Directory Structure
```text
.
├── app.py                         # Streamlit app (loads artifacts + serves recommendations)
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignores venv, caches, secrets, etc.
├── api_key.py                     # (Ignored) TMDB key used by notebooks; don’t commit secrets
├── artifacts/                     # Trained models + configs used by the app
│   ├── svd_model.pkl              # Trained Surprise SVD model (collaborative filtering)
│   ├── svd_metadata.pkl           # Training metadata (metrics + params)
│   ├── hybrid_config.pkl          # Optional config describing hybrid defaults
│   ├── explanation_config.pkl     # Optional config describing explanation versioning
│   └── content/                   # Content-based model artifacts
│       ├── movie_embeddings.npy   # Precomputed embeddings (movies × 384)
│       ├── embedding_metadata.pkl # Metadata about embedding generation (model, dim, etc.)
│       └── enriched_movies_with_embeddings.pkl # Legacy export (currently not required by app)
├── cache/
│   └── tmdb_cache.json            # TMDB enrichment cache generated by notebooks
├── data/
│   ├── ml-latest-small/           # MovieLens “latest small” dataset
│   └── processed/                 # Processed/enriched datasets used by the app
├── movie_recommender/             # Jupyter notebooks (experiments + training pipeline)
└── venv/                          # Local virtualenv (generated; not part of the product)
```

## Installation
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Notebook dependencies (optional)
The notebooks use extra packages that are **not required** to run the Streamlit app. Install them only if you plan to run the Jupyter pipeline:
```bash
pip install -r requirements_notebooks.txt
```

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
This repo is deployable on Streamlit Cloud without committing any API keys.

1. Push the project to GitHub (include `app.py`, `requirements.txt`, `data/`, and `artifacts/`).
2. In Streamlit Cloud, click **New app** and select:
   - Repository + branch
   - Main file path: `movie_recommedation_system/app.py` (or `app.py` if this project is its own repo)
3. This project includes `runtime.txt` to pin the Python version on Streamlit Cloud (avoids Python 3.14 build issues with `scikit-surprise`).
4. (Optional) If you call TMDB APIs from the app, add the key using Streamlit **Secrets**:
   - App → Settings → Secrets:
     ```toml
     TMDB_API_KEY = "your_key_here"
     ```

If you want to **regenerate artifacts** (embeddings / SVD model), run the notebooks in order:
1. `movie_recommender/01_data_preparation.ipynb`
2. `movie_recommender/02_Content_Based_Engine.ipynb`
3. `movie_recommender/03_Collaborative_FIiltering_Engine.ipynb`
4. `movie_recommender/04_Hybird.ipynb`
5. `movie_recommender/05_Adding_Explanation.ipynb`

## Features
- Hybrid recommendations (content + collaborative)
- “More Like This” mode using embeddings + cosine similarity
- Streamlit UI with posters and explanations
- Cached TMDB enrichment (for faster data prep)

## Tech Stack
- Python 3
- Streamlit (UI)
- pandas / NumPy (data)
- scikit-learn (cosine similarity)
- scikit-surprise (SVD collaborative filtering)
- Sentence-Transformers (used in notebooks to build embeddings)
- TMDB API (used in notebooks for metadata/posters)

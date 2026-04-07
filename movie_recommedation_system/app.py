"""
Streamlit Movie Recommender (Hybrid: Content + Collaborative)

What this app does:
- Loads precomputed movie metadata (`data/processed/enriched_movies.csv`)
- Loads precomputed text embeddings (`artifacts/content/movie_embeddings.npy`)
- Loads a collaborative filtering model:
    - Prefer `artifacts/svd_model_light.pkl` (no compilation needed; best for Streamlit Cloud)
    - Fallback to `artifacts/svd_model.pkl` (original Surprise model, good for local dev)
- Computes recommendations in two modes:
    1) "For This User" (mostly collaborative / SVD)
    2) "More Like This" (content similarity boosted, still optionally personalized by userId)

Beginner note: what is `svd_model_light.pkl` and why does it exist?
- The original collaborative model was trained using the `surprise` (scikit-surprise) library and saved as
  `artifacts/svd_model.pkl`.
- Streamlit Cloud often runs very new Python versions, and compiling `scikit-surprise` can fail there.
- To make deployment easy, we export only the *numbers needed for prediction* into `artifacts/svd_model_light.pkl`.

How `svd_model_light.pkl` was created (conceptually):
1) Load the trained Surprise model from `artifacts/svd_model.pkl`
2) Copy out its learned parameters:
   - `global_mean` (average rating)
   - `bu` (user bias), `bi` (movie bias)
   - `pu` (user vectors), `qi` (movie vectors)
   - mappings from raw IDs (userId/movieId) → internal indices
3) Save that as a plain Python dict with pickle

Result:
- The app can compute the same predicted ratings *without* importing/compiling `surprise`.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Movie Recommender", layout="wide", page_icon="🎥")

st.title("🎥 Smart Movie Recommender")
st.markdown("Hybrid Content + Collaborative Recommendations with Explanations")

# --- Collaborative model helpers (no scikit-surprise required in production) ---
def svd_light_predict_est(svd_light, user_id, movie_id):
    """
    Predict rating using a lightweight export of a Surprise SVD model.
    `svd_light` is a dict saved in `artifacts/svd_model_light.pkl`.

    Simple intuition (what the math is doing):
    - Start with the "average rating" across the whole dataset (global_mean).
    - Add a small adjustment for the user:
        bu[u] > 0 means this user tends to give higher ratings than average,
        bu[u] < 0 means this user tends to give lower ratings than average.
    - Add a small adjustment for the movie:
        bi[i] > 0 means this movie is generally liked by everyone,
        bi[i] < 0 means this movie is generally disliked.
    - Add a personalization term (dot product):
        pu[u] and qi[i] are learned vectors; their dot product is high when
        this user's tastes match this movie's "style".

    In one line (simplified):
      predicted_rating = global_mean + user_bias + movie_bias + taste_match

    Cold-start behavior (unseen user/movie):
    - If a userId or movieId was not in the training ratings, we can't look up its learned
      bias/vector, so we safely fall back to whatever parts are available (often just global_mean,
      or global_mean + one bias).
    """
    global_mean = float(svd_light["global_mean"])
    raw2inner_user = svd_light["raw2inner_user"]
    raw2inner_item = svd_light["raw2inner_item"]

    bu = svd_light["bu"]
    bi = svd_light["bi"]
    pu = svd_light["pu"]
    qi = svd_light["qi"]

    u = raw2inner_user.get(int(user_id))
    i = raw2inner_item.get(int(movie_id))

    est = global_mean
    if u is not None:
        est += float(bu[u])
    if i is not None:
        est += float(bi[i])
    if u is not None and i is not None:
        est += float(np.dot(pu[u], qi[i]))

    min_r, max_r = svd_light.get("rating_scale", (0.5, 5.0))
    return float(np.clip(est, min_r, max_r))


# Resolve files relative to this project folder (important for monorepo deployments).
# On Streamlit Cloud, the working directory is typically the repository root, so using
# relative paths like "data/..." can break when the app lives inside a subfolder.
PROJECT_DIR = Path(__file__).resolve().parent

def project_path(*parts: str) -> str:
    """Build a filesystem path relative to `movie_recommedation_system/`."""
    return str(PROJECT_DIR.joinpath(*parts))


# LOAD COMPONENTS
@st.cache_resource
def load_all():
    """
    Load all data and model artifacts once per server process.

    `@st.cache_resource` keeps these heavy objects in memory between reruns:
    - enriched movies dataframe
    - embedding matrix
    - collaborative model (light export preferred)
    - ratings dataframe (to filter already-rated movies)
    """
    enriched = pd.read_csv(project_path("data", "processed", "enriched_movies.csv"))
    
    # Content-based embeddings: one vector per movie row in `enriched_movies`.
    embeddings_path_candidates = [
        project_path("artifacts", "content", "movie_embeddings.npy"),
        project_path("data", "processed", "movie_embeddings.npy"),  # backward-compat fallback
    ]
    embeddings_path = next((p for p in embeddings_path_candidates if os.path.exists(p)), None)
    if embeddings_path is None:
        raise FileNotFoundError(
            "movie embeddings not found. Expected one of: "
            + ", ".join(embeddings_path_candidates)
        )
    embeddings = np.load(embeddings_path)
    
    # Prefer a lightweight collaborative model export so the app can run on Streamlit Cloud
    # without compiling `scikit-surprise`. Fallback to the original Surprise pickle locally.
    svd_model = None
    svd_light_path = project_path("artifacts", "svd_model_light.pkl")
    if os.path.exists(svd_light_path):
        with open(svd_light_path, "rb") as f:
            svd_model = pickle.load(f)
    else:
        with open(project_path("artifacts", "svd_model.pkl"), "rb") as f:
            svd_model = pickle.load(f)
    
    # MovieLens ratings are used only to avoid recommending movies a user already rated.
    ratings = pd.read_csv(project_path("data", "ml-latest-small", "ratings.csv"))
    
    return enriched, embeddings, svd_model, ratings

enriched_movies, movie_embeddings, svd_model, ratings = load_all()

# POSTER FETCHER
def fetch_poster(poster_path):
    """
    Fetch a movie poster image using TMDB's public image CDN.

    Note: This does NOT require a TMDB API key. (TMDB API key is only needed when you
    call TMDB API endpoints to look up metadata.)
    """
    if not poster_path or pd.isna(poster_path):
        return None
    try:
        url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except:
        pass
    return None

#  HYBRID RECOMMENDER FUNCTION 
def get_recommendations(user_id, seed_title=None, top_n=10, alpha=0.6):
    """
    Compute top-N recommendations.

    Two modes:
    - Collaborative-focused (seed_title=None):
        hybrid_score ~= collaborative score
    - Content-boost "More Like This" (seed_title provided):
        hybrid_score is heavily influenced by content similarity (cosine similarity)

    `alpha` controls blending:
      hybrid_score = alpha * content_score + (1 - alpha) * collab_norm
    """
    
    # Movies already rated by this user should not be recommended again.
    user_rated = ratings[ratings['userId'] == user_id]['movieId'].unique()
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Determine effective alpha based on mode.
    effective_alpha = alpha


    # Seed embedding lookup:
    # - Streamlit selectbox provides exact title strings.
    # - We do an exact match first, then a safe substring match (regex=False) because
    #   movie titles often contain parentheses like "(2005)" which break regex matching.
    seed_embedding = None
    if seed_title:
        # `str.contains` uses regex by default; movie titles often contain parentheses.
        # Prefer exact match (selectbox supplies exact titles), then fallback to safe substring match.
        normalized_seed = str(seed_title).strip().lower()
        exact_mask = enriched_movies["title"].astype(str).str.strip().str.lower().eq(normalized_seed)
        if exact_mask.any():
            idx = int(np.flatnonzero(exact_mask.to_numpy())[0])
            seed_embedding = movie_embeddings[idx]
        else:
            contains_mask = enriched_movies["title"].astype(str).str.contains(
                seed_title, case=False, na=False, regex=False
            )
            if contains_mask.any():
                idx = int(np.flatnonzero(contains_mask.to_numpy())[0])
                seed_embedding = movie_embeddings[idx]
            else:
                st.error(f"Seed movie not found: '{seed_title}'. Try selecting a different title.")
                return []

    # Score every candidate movie in the dataset (excluding already-rated ones), then sort.
    results = []
    for i in range(len(enriched_movies)):              
        mid = enriched_movies.iloc[i]['movieId']
        if mid in user_rated:
            continue
        
        row = enriched_movies.iloc[i]
        
        # 1) Collaborative score (SVD predicted rating)
        if isinstance(svd_model, dict) and svd_model.get("type") == "surprise_svd_light_v1":
            est = svd_light_predict_est(svd_model, user_id, mid)
        else:
            pred = svd_model.predict(user_id, mid)
            est = float(pred.est)

        # Normalize to roughly 0–1 so it can be blended with cosine similarity.
        collab_norm = float(np.clip(est / 5.0, 0.0, 1.0))
        
        # 2) Content score (cosine similarity between seed embedding and candidate embedding)
        content_score = 0.0
        if seed_embedding is not None:
            sim = cosine_similarity([seed_embedding], [movie_embeddings[i]])[0][0]
            content_score = float(sim)
        
        # 3) Hybrid blend (content + collaborative)
        hybrid_score = effective_alpha * content_score + (1 - effective_alpha) * collab_norm
        
        # Friendly explanation
        explain = f"Hybrid Score: {hybrid_score:.1%}"
        if seed_title and content_score > 0.75:
            explain += f" • Very strong semantic match to '{seed_title}' ({content_score:.1%})"
        else:
            explain += f" • Predicted rating: {est:.2f}/5"
        
        results.append({
            'title': row['title'],
            'hybrid_score': hybrid_score,
            'collab_pred': est,
            'content_score': content_score,
            'poster_path': row.get('poster_path'),
            'overview': "" if pd.isna(row.get('overview', '')) else str(row.get('overview', '')),
            'explanation': explain
        })
    
    # Sort and return top-N
    results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return results[:top_n]

# SIDEBAR 
st.sidebar.header("🎛️ Controls")

# The app always asks for a userId because:
# - In collaborative mode, userId is required.
# - In content-boost mode, userId can still slightly personalize results via the hybrid blend.
user_id = st.sidebar.number_input("Select User ID", min_value=1, max_value=610, value=42, step=1)

mode = st.sidebar.radio("Mode", 
    ["For This User (Collaborative Focus)", 
     "More Like This Movie (Content Boost)"])

# `alpha` is the content weight used inside `get_recommendations`.
# In "More Like This" mode, we boost alpha to at least 0.92 so results feel strongly related.
alpha = st.sidebar.slider("Content Weight (α)", 0.0, 1.0, 0.6, 0.05)
top_n = st.sidebar.slider("Number of Recommendations", 5, 15, 8)

# MAIN UI
if mode == "For This User (Collaborative Focus)":
    seed_title = None
    st.subheader(f"🎯 Recommendations tailored for User **{user_id}**")
else:
    st.subheader("🔍 More Like This")
    # The selectbox supplies an exact title string from the dataset,
    # which makes seed lookup deterministic.
    seed_title = st.selectbox("Choose a seed movie", 
                             sorted(enriched_movies['title'].unique()), 
                             index=0)

if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
    with st.spinner("Analyzing your taste and finding great movies..."):
        recs = get_recommendations(user_id, seed_title, top_n, alpha)
    
    # Display recommendations as cards.
    cols = st.columns(4)
    for i, rec in enumerate(recs):
        col = cols[i % 4]
        with col:
            poster = fetch_poster(rec['poster_path'])
            if poster:
                st.image(poster, width=300)   
            else:
                st.image("https://via.placeholder.com/300x450.png?text=No+Image", use_column_width=True)
            
            st.subheader(rec['title'])
            st.metric("Hybrid Score", f"{rec['hybrid_score']:.1%}")
            st.caption(f"Predicted Rating: **{rec['collab_pred']:.2f}/5**")
            
            with st.expander("💡 Why this movie?"):
                st.write(rec['explanation'])
                overview = rec.get("overview") or ""
                if overview:
                    overview = str(overview)
                    st.write("**Plot:**", overview[:220] + "..." if len(overview) > 220 else overview)

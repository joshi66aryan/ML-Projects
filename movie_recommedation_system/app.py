import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
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


# LOAD COMPONENTS
@st.cache_resource
def load_all():
    # Paths relative to root
    enriched = pd.read_csv('data/processed/enriched_movies.csv')
    
    embeddings_path_candidates = [
        'artifacts/content/movie_embeddings.npy',
        'data/processed/movie_embeddings.npy',  # backward-compat fallback
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
    svd_light_path = "artifacts/svd_model_light.pkl"
    if os.path.exists(svd_light_path):
        with open(svd_light_path, "rb") as f:
            svd_model = pickle.load(f)
    else:
        with open("artifacts/svd_model.pkl", "rb") as f:
            svd_model = pickle.load(f)
    
    ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
    
    return enriched, embeddings, svd_model, ratings

enriched_movies, movie_embeddings, svd_model, ratings = load_all()

# POSTER FETCHER
def fetch_poster(poster_path):
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
    """FINAL VERSION: Full dataset + strong content boost for 'More Like This'"""
    
    user_rated = ratings[ratings['userId'] == user_id]['movieId'].unique()
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Determine effective alpha based on mode.
    effective_alpha = alpha
    if seed_title:                                      # "More Like This" mode
        # Keep the app content-heavy by default, but still allow alpha=1.0 for pure content.
        effective_alpha = max(alpha, 0.92)
        if effective_alpha != alpha:
            st.info("🔥 Content mode active — alpha boosted to 0.92 for strong similarity")
    
    # Get the seed embeddings if needed
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

    results = []
    for i in range(len(enriched_movies)):              
        mid = enriched_movies.iloc[i]['movieId']
        if mid in user_rated:
            continue
        
        row = enriched_movies.iloc[i]
        
        # 1. Collaborative score (SVD)
        if isinstance(svd_model, dict) and svd_model.get("type") == "surprise_svd_light_v1":
            est = svd_light_predict_est(svd_model, user_id, mid)
        else:
            pred = svd_model.predict(user_id, mid)
            est = float(pred.est)
        collab_norm = float(np.clip(est / 5.0, 0.0, 1.0))
        
        # 2. Content score
        content_score = 0.0
        if seed_embedding is not None:
            sim = cosine_similarity([seed_embedding], [movie_embeddings[i]])[0][0]
            content_score = float(sim)
        
        # 3. Hybrid score (now correctly content-heavy when seed is given)
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

user_id = st.sidebar.number_input("Select User ID", min_value=1, max_value=610, value=42, step=1)

mode = st.sidebar.radio("Mode", 
    ["For This User (Collaborative Focus)", 
     "More Like This Movie (Content Boost)"])

alpha = st.sidebar.slider("Content Weight (α)", 0.0, 1.0, 0.6, 0.05)
top_n = st.sidebar.slider("Number of Recommendations", 5, 15, 8)

# MAIN UI
if mode == "For This User (Collaborative Focus)":
    seed_title = None
    st.subheader(f"🎯 Recommendations tailored for User **{user_id}**")
else:
    st.subheader("🔍 More Like This")
    seed_title = st.selectbox("Choose a seed movie", 
                             sorted(enriched_movies['title'].unique()), 
                             index=0)

if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
    with st.spinner("Analyzing your taste and finding great movies..."):
        recs = get_recommendations(user_id, seed_title, top_n, alpha)
    
    # Display as beautiful cards
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

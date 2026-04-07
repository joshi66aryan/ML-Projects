# data/ml-latest-small/

This is the **MovieLens “latest small”** dataset.

It is used mainly to:
- Train the collaborative filtering model (SVD)
- Provide base movie identifiers (`movieId`) that are later enriched with TMDB metadata

## Files
- `README.txt`
  - Original dataset documentation (from MovieLens).
- `movies.csv`
  - Movie metadata (movieId, title, genres).
- `ratings.csv`
  - User ratings (userId, movieId, rating, timestamp).
- `links.csv`
  - Links from MovieLens IDs to external IDs (like TMDB).
- `tags.csv`
  - User-generated tags (optional signal; mainly used during exploration).


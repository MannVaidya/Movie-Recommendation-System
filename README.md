# Movie Recommender — Streamlit + TMDB

A Streamlit frontend inspired by a six-column movie gallery. TMDB API provides title-search results, trending and popular movies, and poster images. Your notebook's TF-IDF matrix and cosine similarity produce actual content-based recommendations.

## Folder layout

```text
Movie_Recommendation_System/
├── app.py
├── models/
│   ├── df.pkl                    # Export from notebook
│   └── tfidf_matrix.pkl          # Export from notebook
├── notebooks/
│   └── movie_recommendation.ipynb  # Optional, your own notebook
├── .streamlit/
│   └── config.toml
├── .env                        # Make from .env.example, don't commit
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. In your original notebook, run the training cells and then export:

   ```python
   import pickle
   df.to_pickle('df.pkl')
   with open('tfidf_matrix.pkl', 'wb') as file:
       pickle.dump(tfidf_matrix, file)
   ```

   Copy both files to `models/`. **Re-export both together**; model and dataset row order must match.
2. Copy `.env.example` to `.env` and add your real TMDB API v3 key:

   ```ini
   TMDB_API_KEY=YOUR_REAL_TMDB_API_KEY
   ```

   The TMDB **API Read Access Token** is also supported, using `TMDB_API_READ_ACCESS_TOKEN` instead. Do not put quotes around your API key unless needed.
3. Open a VS Code terminal inside the project folder and run:

   ```powershell
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

4. Search `toy story` to see a poster grid. Choose *Toy Story* under **Suggestions** or click **Find similar** on a poster to see cosine-similarity recommendations.

## What gets recommended?

- **Results** shows a live TMDB text search (or trending/popular movies when the search field is empty). These are not recommendations from your ML model.
- **Recommended for: ...** is powered by your TF-IDF model and cosine similarity.
- A film available on TMDB may not exist in your model's dataset, particularly if released after your dataset was collected. Those movies can appear in search, but the model can recommend only for titles it was trained on.
- The notebook currently discards TMDB IDs. This app looks up a title to find its poster. Movies with duplicate titles may occasionally show an incorrect poster. Retaining the original `id` in your dataset can fix that in a future version.

## Troubleshooting

- `Cannot find df.pkl` — copy your pickle files to `models/` (or directly alongside `app.py`).
- `Movie rows and TF-IDF matrix rows do not match` — regenerate both pickle files in the same notebook run.
- `TMDB API returned HTTP 401` — your API key / read token is wrong or expired.
- `No poster` — TMDB returned no poster for this movie, or your API key/network isn't working.
- If you're using Python 3.13 or a newer version than you used for training, and pickle loading fails due to scikit-learn version changes, use the same scikit-learn and Python versions as your training environment or re-export the model.

Movie metadata/images from TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.

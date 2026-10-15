"""TMDB poster browser + a TF-IDF/cosine-similarity movie recommender.

Uses the pickle artifacts exported by the accompanying Jupyter notebook:
  - df.pkl
  - tfidf_matrix.pkl

Put the files in ./models/ (preferred) or alongside app.py.
API key: set TMDB_API_KEY in .env, or use Streamlit secrets.
"""

from __future__ import annotations

import html
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
TMDB_API_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"
PAGE_SIZE = 12

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { max-width: 1650px; padding-top: 2.7rem; }
      [data-testid="stSidebar"] { background-color: #23262f; }
      [data-testid="stImage"] img {
        aspect-ratio: 2 / 3;
        object-fit: cover;
        border-radius: 10px;
      }
      .movie-title {
        font-size: 0.89rem;
        font-weight: 650;
        margin-top: 0.42rem;
        min-height: 2.4em;
        line-height: 1.3;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
      }
      .poster-missing {
        aspect-ratio: 2 / 3;
        border-radius: 10px;
        background: #2a2d38;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #a5a9b5;
        font-size: 0.9rem;
        padding: 1rem;
      }
      .app-subtitle { color: #a8abb5; margin-top: -0.65rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading your recommendation model...")
def load_model():
    """Load the two pickle files produced by the user's existing notebook."""

    def locate(name: str) -> Path:
        for path in (BASE_DIR / "models" / name, BASE_DIR / name):
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Cannot find {name}. Copy it into '{BASE_DIR / 'models'}' "
            "or next to app.py, then restart Streamlit."
        )

    movies = pd.read_pickle(locate("df.pkl")).reset_index(drop=True)
    with locate("tfidf_matrix.pkl").open("rb") as file:
        matrix = pickle.load(file)

    if "title" not in movies.columns:
        raise ValueError("df.pkl must have a 'title' column.")
    if len(movies) != matrix.shape[0]:
        raise ValueError(
            "Movie rows and TF-IDF matrix rows do not match. "
            "Re-export df.pkl and tfidf_matrix.pkl from the same notebook run."
        )
    movies["title"] = movies["title"].fillna("").astype(str)
    return movies, matrix


def read_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


@st.cache_data(ttl=60 * 60, show_spinner=False)
def tmdb_get(path: str, api_key: str, read_token: str, params: dict | None = None):
    """Request TMDB data; return a JSON dictionary or raise a friendly error."""
    if not api_key and not read_token:
        return {}
    headers = {"accept": "application/json"}
    query = dict(params or {})
    if read_token:
        headers["Authorization"] = f"Bearer {read_token}"
    else:
        query["api_key"] = api_key
    try:
        response = requests.get(
            f"{TMDB_API_URL}{path}",
            params=query,
            headers=headers,
            timeout=8,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise RuntimeError(
            f"TMDB API returned HTTP {status}. Check your API key/token and internet connection."
        ) from None
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"Could not contact TMDB: {type(exc).__name__}.") from None


def poster_url(movie: dict) -> str | None:
    path = movie.get("poster_path")
    return f"{TMDB_IMAGE_URL}{path}" if path else None


def movie_link(movie: dict) -> str | None:
    movie_id = movie.get("id")
    return f"https://www.themoviedb.org/movie/{movie_id}" if movie_id else None


def search_tmdb(query: str, api_key: str, read_token: str) -> list[dict]:
    if not query.strip():
        return []
    return tmdb_get(
        "/search/movie",
        api_key,
        read_token,
        {"query": query.strip(), "include_adult": "false", "language": "en-US"},
    ).get("results", [])


def featured_tmdb_movies(category: str, api_key: str, read_token: str) -> list[dict]:
    paths = {
        "Trending": "/trending/movie/week",
        "Popular": "/movie/popular",
        "Top Rated": "/movie/top_rated",
        "Now Playing": "/movie/now_playing",
    }
    return tmdb_get(
        paths[category], api_key, read_token, {"language": "en-US"}
    ).get("results", [])


def find_tmdb_movie(title: str, api_key: str, read_token: str) -> dict:
    """Best-effort title lookup; prefer an exact title and available poster."""
    results = search_tmdb(title, api_key, read_token)
    if not results:
        return {"title": title}
    exact = [
        item for item in results
        if str(item.get("title", "")).casefold() == title.casefold()
    ]
    return next(
        (item for item in exact if item.get("poster_path")),
        next((item for item in results if item.get("poster_path")),
             exact[0] if exact else results[0]),
    )


def suggest_titles(query: str, all_titles: list[str], limit: int = 30) -> list[str]:
    query = query.strip().casefold()
    if not query:
        return []
    found = [title for title in all_titles if query in title.casefold()]
    found.sort(key=lambda title: (
        not title.casefold().startswith(query),
        len(title),
        title.casefold(),
    ))
    return found[:limit]


def recommend(title: str, movies: pd.DataFrame, matrix, n: int) -> list[tuple[str, float]]:
    """Content-based recommendations from the user's original TF-IDF matrix."""
    mask = movies["title"].str.casefold() == title.casefold()
    indices = movies.index[mask]
    if len(indices) == 0:
        return []
    source_index = int(indices[0])
    scores = cosine_similarity(matrix[source_index], matrix).ravel()
    ranked_indices = np.argsort(scores)[::-1]

    output: list[tuple[str, float]] = []
    seen = {title.casefold()}
    for index in ranked_indices:
        index = int(index)
        score = float(scores[index])
        if score <= 0:
            break
        candidate = movies.iloc[index]["title"].strip()
        if index == source_index or not candidate or candidate.casefold() in seen:
            continue
        output.append((candidate, score))
        seen.add(candidate.casefold())
        if len(output) >= n:
            break
    return output


def movie_card(movie: dict, key: str, *, score: float | None = None,
               can_recommend: bool = False, known_titles: dict[str, str] | None = None):
    """Show a poster/title card, with either a selection button or TMDB link."""
    title = str(movie.get("title") or "Unknown title")
    poster = poster_url(movie)
    if poster:
        st.image(poster, width="stretch")
    else:
        st.markdown(
            '<div class="poster-missing">🎞️<br>Poster unavailable</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="movie-title" title="{html.escape(title, quote=True)}">'
        f'{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )
    if score is not None:
        st.caption(f"Similarity: {score:.2f}")
    else:
        release_date = str(movie.get("release_date") or "")
        year = release_date[:4] if len(release_date) >= 4 else ""
        vote = movie.get("vote_average")
        score_text = f"⭐ {float(vote):.1f}" if isinstance(vote, (float, int)) and vote > 0 else ""
        st.caption(" · ".join(part for part in (year, score_text) if part) or "Movie")

    if can_recommend:
        if st.button("Find similar", key=key, use_container_width=True):
            matched_title = (known_titles or {}).get(title.casefold())
            if matched_title:
                st.session_state["selected_title"] = matched_title
                st.rerun()
            else:
                st.info("Not in your trained dataset. Choose a title from Suggestions.")
    else:
        link = movie_link(movie)
        if link:
            st.link_button("TMDB details ↗", link, use_container_width=True)


def display_grid(items: list[dict], column_count: int, *, scores=None,
                 selectable: bool = False, known_titles=None, prefix="movie"):
    if not items:
        st.info("No movies found.")
        return
    for start in range(0, len(items), column_count):
        cols = st.columns(column_count, gap="medium")
        for col, i in zip(cols, range(start, min(start + column_count, len(items)))):
            with col:
                movie_card(
                    items[i], key=f"{prefix}-{i}",
                    score=scores[i] if scores else None,
                    can_recommend=selectable,
                    known_titles=known_titles,
                )


def main():
    try:
        movies, matrix = load_model()
    except (FileNotFoundError, ValueError, OSError, ImportError, pickle.UnpicklingError) as exc:
        st.error(str(exc))
        st.info("Export the model files from your notebook, then place them inside the models folder.")
        st.stop()

    api_key = read_secret("TMDB_API_KEY")
    read_token = read_secret("TMDB_API_READ_ACCESS_TOKEN")
    tmdb_ready = bool(api_key or read_token)
    all_titles = sorted(set(t.strip() for t in movies["title"] if t.strip()))
    known_titles = {title.casefold(): title for title in all_titles}
    if "selected_title" not in st.session_state:
        st.session_state["selected_title"] = None

    with st.sidebar:
        st.header("🎬 Menu")
        st.markdown("🏠 **Home**")
        st.divider()
        category = st.selectbox("Home category", ["Trending", "Popular", "Top Rated", "Now Playing"])
        columns = st.slider("Grid columns", 2, 6, 6)
        recommendation_count = st.slider("Recommendations", 3, 12, 6)
        if st.button("🔄 Refresh TMDB data", use_container_width=True):
            tmdb_get.clear()
            st.rerun()
        st.divider()
        st.caption(f"📚 {len(all_titles):,} movies in your trained dataset")
        st.caption("Movie recommendations: TF-IDF + cosine similarity")
        st.caption("Movie posters, searches, ratings: TMDB API")

    st.title("🎬 Movie Recommender")
    st.markdown(
        '<p class="app-subtitle">Search movies, explore posters, and discover recommendations</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    if not tmdb_ready:
        st.warning(
            "TMDB API key missing: add TMDB_API_KEY=your_key to the .env file. "
            "You can still select movies, but posters and TMDB searches need the API key."
        )

    query = st.text_input("Search by movie title", value="toy story", placeholder="e.g. Avatar, Interstellar")
    matches = suggest_titles(query, all_titles)
    choice = st.selectbox(
        "Suggestions (movies in your trained model)",
        ["-- Select a movie --", *matches],
        index=0,
    )
    if choice != "-- Select a movie --":
        st.session_state["selected_title"] = choice

    selected = st.session_state["selected_title"]
    if selected:
        st.divider()
        heading, clear = st.columns([6, 1])
        heading.subheader(f"🍿 Recommended for: {selected}")
        if clear.button("Clear selection", use_container_width=True):
            st.session_state["selected_title"] = None
            st.rerun()

        similar_movies = recommend(selected, movies, matrix, recommendation_count)
        if not similar_movies:
            st.info("No similar movies found in the trained model.")
        elif tmdb_ready:
            with st.spinner("Fetching recommended movie posters..."):
                recommendation_items = []
                for title, _ in similar_movies:
                    try:
                        recommendation_items.append(find_tmdb_movie(title, api_key, read_token))
                    except RuntimeError:
                        recommendation_items.append({"title": title})
            display_grid(
                recommendation_items, columns,
                scores=[score for _, score in similar_movies],
                prefix="recommendation",
            )
        else:
            display_grid(
                [{"title": title} for title, _ in similar_movies], columns,
                scores=[score for _, score in similar_movies],
                prefix="recommendation",
            )

    st.divider()
    st.subheader("Results" if query.strip() else category)
    if tmdb_ready:
        try:
            with st.spinner("Loading TMDB movie posters..."):
                results = (
                    search_tmdb(query, api_key, read_token)
                    if query.strip() else featured_tmdb_movies(category, api_key, read_token)
                )
        except RuntimeError as exc:
            st.error(str(exc))
            results = []
        display_grid(
            results[:PAGE_SIZE], columns,
            selectable=True, known_titles=known_titles, prefix="search",
        )
        if query.strip():
            st.caption("These are TMDB search results. Select a film in your model to generate personalized similar-movie recommendations.")
    else:
        st.info("TMDB API is needed for poster results. Here are matching titles from your model instead:")
        display_grid(
            [{"title": title} for title in matches[:PAGE_SIZE]], columns,
            selectable=True, known_titles=known_titles, prefix="local",
        )

    st.divider()
    st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")


if __name__ == "__main__":
    main()

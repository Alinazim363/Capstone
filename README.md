# SweetSpot

**SweetSpot** is a transit-aware meetup optimizer for New York City. Given two addresses, it uses the MTA subway graph to find the fairest meeting point, where neither person travels significantly longer than the other, then ranks nearby venues by semantic similarity to a natural-language "vibe" query.

**Disclaimer** - SweetSpot only works with addresses located in Manhattan.

**Link to GitHub Repo** - [github.com/Alinazim363/Capstone] (https://github.com/Alinazim363/Capstone/)

> **Live Web App:** [sweetspotcapstone.streamlit.app](https://sweetspotcapstone.streamlit.app/)

---

## How It Works

1. **Geocoding** — Converts two user addresses into coordinates and snaps them to the nearest MTA subway platform.
2. **Isochrone Routing (Dijkstra)** — Runs `nx.single_source_dijkstra_path_length` from both starting platforms to map all reachable stations within 60 minutes.
3. **Fairness Filter** — Keeps only stations where the difference in travel time between both users is ≤ 7 minutes.
4. **Geographic Filter** — Finds Yelp venues within 800 meters of any fair transit hub.
5. **Semantic Ranking (NLP)** — Encodes the user's vibe query with Sentence-RoBERTa and ranks venues by cosine similarity to pre-computed vibe vectors.
6. **A\* Pathfinding** — Generates step-by-step subway itineraries for each user to their chosen venue.

---

## Project Structure

| File | Description |
|---|---|
| `app.py` | **Main terminal application.** Full SweetSpot pipeline — geocoding, Dijkstra isochrones, NLP ranking, and A\* directions. |
| `NetworkGraph.py` | Builds the directed MTA subway graph from GTFS data. Imported by all other modules. |
| `geocoding.py` | Geocodes addresses using Nominatim and snaps coordinates to the nearest subway platform. |
| `frontend.py` | Streamlit web frontend. Mirrors `app.py` logic with an interactive map and visual UI. |
| `IsochroneMidpoint_generation.py` | Standalone script for testing isochrone generation between two hardcoded station IDs. |
| `vibe_vector.ipynb` | One-time Colab notebook used to generate and save the `sweetspot_production_db_v2.json` vibe vectors. Do not re-run. |
| `tests/` | Unit tests, isochrone tests, and NLP vibe search diagnostics. |

---

## Prerequisites

- Python 3.10+
- MTA GTFS data files in a `gtfs/` directory (see below)
- Yelp venue database at `yelp/sweetspot_production_db_v2.json`

### Required GTFS Files

Place the following files inside a `gtfs/` folder in the project root:

```
gtfs/
├── stops.txt
├── trips.txt
├── stop_times.txt
└── transfers.txt
```

MTA GTFS data can be downloaded from [mta.info/developers](https://new.mta.info/developers).

---

## Installation

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd sweetspot
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

> The first run will download the `all-distilroberta-v1` Sentence Transformer model (~330MB). This happens automatically and only once.

---

## Running the App

```bash
python3 app.py
```

You will be prompted for:

```
User 1 Address: 14th St and 6th Ave, New York
User 2 Address: 86th St and Lexington Ave, New York
Describe the vibe: a quiet wine bar for a first date
```

The app will output the **Top 10 venue recommendations** ranked by vibe match, along with transit hub names and commute times for both users. You can then select any result to get full step-by-step A\* subway directions for both users.

---

## Web Frontend

A Streamlit-based frontend with an interactive Folium map is also available. It runs the same backend pipeline with a visual interface.

**To Run locally:**

```bash
streamlit run frontend.py
```

**Or visit the live deployment:**
[https://sweetspotcapstone.streamlit.app/](https://sweetspotcapstone.streamlit.app/)


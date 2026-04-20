# Clustering Emotion: A Computational Exploration of Music and Mood

This project is an end-to-end Python application that explores whether songs can be grouped into meaningful emotional clusters using audio features such as valence, energy, tempo, danceability, acousticness, speechiness, liveness, and instrumentalness.

The system follows the exact spirit of the proposal: it loads a song dataset, normalizes selected features, applies K-means clustering, projects the clusters into two dimensions with PCA, and gives an interpretable summary of each cluster for mood-based analysis.

## Project Features

- Loads a Spotify-style CSV dataset
- Cleans and standardizes audio features
- Uses **K-means clustering** to group songs by similarity
- Uses **silhouette score** to recommend the best number of clusters
- Uses **PCA** to visualize high-dimensional song features in 2D
- Assigns human-readable mood labels to each cluster based on energy and valence
- Lets the user inspect songs inside each emotional cluster
- Exports final clustered results as a CSV file

## Project Structure

```text
music_mood_project/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_songs.csv
├── notebooks/
└── src/
    ├── config.py
    ├── data_utils.py
    ├── clustering.py
    ├── emotion.py
    └── visuals.py
```

## Dataset Requirements

Your CSV should include these columns:

- `track_name`
- `artist`
- `track_genre`
- `popularity`
- `danceability`
- `energy`
- `valence`
- `tempo`
- `acousticness`
- `speechiness`
- `liveness`
- `instrumentalness`

A small example dataset is already included in `data/sample_songs.csv` so the app can run immediately.

## Installation

1. Clone or download the project.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Start the interactive app with:

```bash
streamlit run app.py
```

After that, Streamlit will open the project in your browser.

##Example Usage
- Load the dataset automatically when the app starts
- Adjust the number of clusters using the slider
- View how songs are grouped in the PCA visualization
- Observe how cluster characteristics change based on parameters

## Example Workflow

1. Launch the app.
2. Use the included sample dataset or upload your own CSV.
3. Let the system choose the best number of clusters automatically, or manually choose `k`.
4. View the PCA plot to see how songs are grouped.
5. Review the cluster summary table.
6. Inspect the songs inside each cluster.
7. Download the final clustered dataset.

## Suggested Analysis Questions

- Do high-valence, high-energy songs group together consistently?
- Are low-energy songs more likely to appear in reflective or calm clusters?
- How well do the clusters align with psychological models such as Russell’s Circumplex Model?
- Does changing the number of clusters improve interpretability?

## Future Improvements

- Add more clustering algorithms such as hierarchical clustering or DBSCAN
- Include richer emotional labeling strategies
- Add direct Spotify API integration
- Build playlist recommendations from cluster membership
- Compare clustering results against human mood labels


import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# import umap
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go


DATASET            = "/Users/bytedance/Downloads/ChatQilin-zh-BZ1K/train.50K.src_tgt.text.json"
DATASET_EMBEDDINGS = "/Users/bytedance/Downloads/ChatQilin-zh-BZ1K/train.50K.src_tgt.emb.json"

OUTPUT_PATH        = "/Users/bytedance/Downloads/ChatQilin-zh-BZ1K/train.50K.src_tgt.vis.json"


# Load dataset

with open(DATASET, "r") as f_t, open(DATASET_EMBEDDINGS, "r") as f_e:
    texts = json.load(f_t)
    embeddings = np.array(json.load(f_e))

    assert len(texts) == len(embeddings)

# Assume all embeddings have unit length
# Then Euclidean distance ||x - y|| = 2(1 - cos(x, y))

assert np.isclose(np.linalg.norm(embeddings, axis=-1), 1.0).all()

# Dimensional reduction

transform = umap.UMAP(n_neighbors=500, min_dist=0.1, n_components=3, metric="euclidean", low_memory=False)
embeddings_umap = transform.fit_transform(embeddings)

# K-Means classification

ks = []
scores = []
labels = []
for k in tqdm(range(20, 21)):
    km = KMeans(k, n_init=10).fit(embeddings)

    ks.append(k)
    labels.append(km.labels_)
    scores.append(silhouette_score(embeddings, km.labels_, metric="euclidean"))

fig = go.Figure(data=[go.Scatter(x=ks, y=scores, mode="lines+markers")])
fig.show()

# Use optimal K as pseudo labels

labels = labels[np.argmax(scores)]

# 3D Figure
fig = go.Figure(data=[
    go.Scatter3d(
        x=embeddings_umap[:, 0], y=embeddings_umap[:, 1], z=embeddings_umap[:, 2],
        mode="markers",
        marker=dict(
            size=1.5,
            opacity=0.8,
            color=labels
        )
    )
])

fig.show()

# Save visualizer to HTML

visualizer_data = json.dumps({"text": texts, "embedding": embeddings_umap.tolist(), "color": labels.tolist()})

with open(OUTPUT_PATH, "w") as f:
    f.write(visualizer_data)
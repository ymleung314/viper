import numpy as np
from sklearn.cluster import KMeans

INPUT_FILE = "/data/ymliang/sample-1.fbin"
N_CLUSTERS = 2
RANDOM_SEED = 0


def main():
    print("Extracting vectors...")
    with open(INPUT_FILE, "rb") as inf:
        n_vec = np.fromfile(inf, dtype=np.int32, count=1)[0]
        vec_dim = np.fromfile(inf, dtype=np.int32, count=1)[0]
        vecs_raw = np.fromfile(inf, dtype=np.float32, count=-1)

    vecs = vecs_raw.reshape(n_vec, vec_dim)
    print(f"Dataset shape: {vecs.shape}")

    print("Running KMeans...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    kmeans.fit(vecs)
    print(kmeans.cluster_centers_)


if __name__ == "__main__":
    main()

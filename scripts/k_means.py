import struct

import numpy as np
from sklearn.cluster import KMeans

INPUT_FILE = "/data/ymliang/sample-1.fbin"
OUTPUT_FILE = "/data/ymliang/means-5.fbin"
N_CLUSTERS = 5
RANDOM_SEED = 0


def main():
    print(f"Extracting vectors from {INPUT_FILE}")
    with open(INPUT_FILE, "rb") as inf:
        n_vec = np.fromfile(inf, dtype=np.int32, count=1)[0]
        vec_dim = np.fromfile(inf, dtype=np.int32, count=1)[0]
        vecs_raw = np.fromfile(inf, dtype=np.float32, count=-1)

    vecs = vecs_raw.reshape(n_vec, vec_dim)
    print(f"Dataset shape: {vecs.shape}")

    print("Running KMeans...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED)
    kmeans.fit(vecs)

    print(f"Saving means to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "wb") as outf:
        outf.write(struct.pack("<ii", N_CLUSTERS, vec_dim))

        for mean in kmeans.cluster_centers_:
            np.array(mean, np.float32).tofile(outf)


if __name__ == "__main__":
    main()

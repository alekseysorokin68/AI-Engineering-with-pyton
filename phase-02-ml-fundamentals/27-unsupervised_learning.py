"""
Unsupervised Learning: K-Means & DBSCAN from scratch (no sklearn)
"""

import random
import math

random.seed(42)


# ──────────────────────────── utilities ────────────────────────────

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def generate_clusters(n_per_cluster=50, n_clusters=3):
    """Generate 3 Gaussian blobs with known centers."""
    centers = [
        [random.uniform(-5, -1), random.uniform(-5, -1)],
        [random.uniform(3, 7),   random.uniform(3, 7)],
        [random.uniform(-2, 2),  random.uniform(4, 8)],
    ]
    points, labels = [], []
    spread = 0.8
    for ci, center in enumerate(centers):
        for _ in range(n_per_cluster):
            p = [center[i] + random.gauss(0, spread) for i in range(2)]
            points.append(p)
            labels.append(ci)
    return points, labels


# ──────────────────────────── K-Means ─────────────────────────────

class KMeans:
    """K-Means clustering from scratch."""

    def __init__(self, k=3, max_iter=100, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = []
        self.labels = []

    def fit(self, X):
        n = len(X)
        idxs = random.sample(range(n), self.k)
        self.centroids = [list(X[i]) for i in idxs]

        for _ in range(self.max_iter):
            # assign
            self.labels = [self._closest(p) for p in X]

            # update centroids
            new_centers = []
            for c in range(self.k):
                members = [X[i] for i in range(n) if self.labels[i] == c]
                if not members:
                    new_centers.append(self.centroids[c])
                else:
                    dim = len(members[0])
                    new_centers.append([sum(m[d] for m in members) / len(members) for d in range(dim)])

            shift = max(euclidean(new_centers[i], self.centroids[i]) for i in range(self.k))
            self.centroids = new_centers
            if shift < self.tol:
                break

    def predict(self, X):
        return [self._closest(p) for p in X]

    def inertia(self, X):
        return sum(euclidean(X[i], self.centroids[self.labels[i]]) ** 2 for i in range(len(X)))

    def _closest(self, p):
        return min(range(self.k), key=lambda c: euclidean(p, self.centroids[c]))


# ──────────────────────────── DBSCAN ──────────────────────────────

class DBSCAN:
    """DBSCAN clustering from scratch."""

    UNDEFINED = -2
    NOISE = -1

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = []

    def fit(self, X):
        n = len(X)
        labels = [self.UNDEFINED] * n
        cluster_id = 0

        for i in range(n):
            if labels[i] != self.UNDEFINED:
                continue
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                labels[i] = self.NOISE
                continue
            labels[i] = cluster_id
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                if labels[q] == self.UNDEFINED:
                    labels[q] = cluster_id
                    new_neighbors = self._region_query(X, q)
                    if len(new_neighbors) >= self.min_samples:
                        for nb in new_neighbors:
                            if labels[nb] in (self.UNDEFINED, self.NOISE):
                                if labels[nb] == self.UNDEFINED:
                                    seed_set.append(nb)
                elif labels[q] == self.NOISE:
                    labels[q] = cluster_id
                j += 1
            cluster_id += 1

        self.labels = labels

    def predict(self, X):
        return [self._assign(p, X) for p in X]

    def _region_query(self, X, idx):
        return [j for j in range(len(X)) if euclidean(X[idx], X[j]) <= self.eps]

    def _assign(self, p, X):
        dists = [euclidean(p, x) for x in X]
        nearest = min(range(len(dists)), key=lambda i: dists[i])
        return self.labels[nearest]


# ──────────────────────────── Metrics ─────────────────────────────

def silhouette_score(X, labels):
    """Average silhouette coefficient (higher = better)."""
    n = len(X)
    unique = set(labels)
    unique.discard(-1)
    if len(unique) < 2:
        return 0.0

    total, count = 0.0, 0
    for i in range(n):
        li = labels[i]
        if li == -1:
            continue
        same = [j for j in range(n) if labels[j] == li and j != i]
        if not same:
            continue
        a = sum(euclidean(X[i], X[j]) for j in same) / len(same)

        b = float("inf")
        for c in unique:
            if c == li:
                continue
            others = [j for j in range(n) if labels[j] == c]
            if others:
                avg = sum(euclidean(X[i], X[j]) for j in others) / len(others)
                b = min(b, avg)

        if b == float("inf"):
            continue
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        total += s
        count += 1

    return total / count if count else 0.0


# ──────────────────────────── Demo ────────────────────────────────

def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    X, true_labels = generate_clusters(50, 3)

    # ── Demo 1: K-Means with different K ──
    print_separator("Demo 1: K-Means — разные значения K")
    for k in [2, 3, 5]:
        km = KMeans(k=k)
        km.fit(X)
        sil = silhouette_score(X, km.labels)
        print(f"\nK={k}  |  Inertia: {km.inertia(X):.2f}  |  Silhouette: {sil:.4f}")
        for c in range(k):
            cnt = km.labels.count(c)
            print(f"  Cluster {c}: {cnt} points, center={[round(v,2) for v in km.centroids[c]]}")

    # ── Demo 2: Elbow method ──
    print_separator("Demo 2: Elbow Method (инерция по K)")
    inertias = []
    for k in range(1, 11):
        km = KMeans(k=k)
        km.fit(X)
        inertias.append(km.inertia(X))
        print(f"  K={k:>2d}  ->  Inertia = {inertias[-1]:>10.2f}")

    print("\n  Elbow ожидается вокруг K=3 (резкое замедление падения инерции).")

    # ── Demo 3: DBSCAN with different params ──
    print_separator("Demo 3: DBSCAN — разные параметры")
    configs = [
        (0.5, 5, "_eps=0.5, min_samples=5"),
        (1.0, 5, "_eps=1.0, min_samples=5"),
        (1.5, 5, "_eps=1.5, min_samples=5"),
        (0.8, 3, "eps=0.8, min_samples=3"),
    ]
    for eps, ms, label in configs:
        db = DBSCAN(eps=eps, min_samples=ms)
        db.fit(X)
        n_clusters = len(set(db.labels) - {-1})
        n_noise = db.labels.count(-1)
        sil = silhouette_score(X, db.labels)
        print(f"\n  {label}")
        print(f"    Clusters found: {n_clusters}  |  Noise points: {n_noise}  |  Silhouette: {sil:.4f}")

    # ── Demo 4: K-Means vs DBSCAN ──
    print_separator("Demo 4: K-Means vs DBSCAN — сравнение")

    km = KMeans(k=3)
    km.fit(X)
    km_sil = silhouette_score(X, km.labels)

    db = DBSCAN(eps=1.0, min_samples=5)
    db.fit(X)
    db_sil = silhouette_score(X, db.labels)
    db_clusters = len(set(db.labels) - {-1})

    print(f"\n  {'Metric':<22} {'K-Means (k=3)':>15} {'DBSCAN':>15}")
    print(f"  {'-'*52}")
    print(f"  {'Inertia':<22} {km.inertia(X):>15.2f} {'N/A':>15}")
    print(f"  {'Silhouette':<22} {km_sil:>15.4f} {db_sil:>15.4f}")
    print(f"  {'Clusters':<22} {'3':>15} {db_clusters:>15}")
    print(f"  {'Noise points':<22} {'0':>15} {db.labels.count(-1):>15}")

    print("\n  Вывод:")
    print("  - K-Means хорошо работает когда кластеры сферические и K известен.")
    print("  - DBSCAN находит кластеры произвольной формы и не требует задания K.")
    print("  - DBSCAN может выделять шумовые точки (label = -1).")


if __name__ == "__main__":
    main()

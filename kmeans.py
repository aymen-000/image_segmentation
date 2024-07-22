import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=200, plot_phases=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_phases = plot_phases
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def predict(self, x):
        self.x = x
        self.n_samples, self.n_features = x.shape
        random_sample_idx = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.x[i] for i in random_sample_idx]
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_phases:
                self.plot()
            old_centers = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self._conv(old_centers, self.centroids):
                break
            if self.plot_phases:
                self.plot()
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for clu_idx, cluster in enumerate(clusters):
            for sample in cluster:
                labels[sample] = clu_idx
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.x):
            center_idx = self._closest_centroid(sample, centroids)
            clusters[center_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [self._get_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_distance(self, sample, point, norm='euclidean'):
        if norm == 'euclidean':
            return np.linalg.norm(sample - point)
        else:
            raise ValueError(f"Unsupported norm type: {norm}")

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _conv(self, center_old, center_new):
        distances = [np.linalg.norm(center_old[i] - center_new[i]) for i in range(self.k)]
        return np.sum(distances, axis=0) <= 0.01

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.x[index].T
            ax.scatter(*point)
        for point in self.centroids:
            ax.scatter(*point, marker="x", color='black', linewidth=2)
        plt.show()

    def cent(self):
        return self.centroids
import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float


def kmeans(features, k, num_iters=100):
    N, D = features.shape
    assert N >= k, 'Number of clusters cannot be greater than number of points'
    index = np.random.choice(N, size=k, replace=False)
    centers = features[index]
    assignments = np.zeros(N, dtype=np.uint32)
    for n in range(num_iters):
        prev_assignments = assignments.copy()
        for i, point in enumerate(features):
            distances = np.linalg.norm(point - centers, axis=1)
            assignments[i] = np.argmin(distances)
        for j in range(k):
            cluster_points = features[assignments == j]
            centers[j] = cluster_points.mean(axis=0)
        if (assignments == prev_assignments).all():
            break
    return assignments


def kmeans_fast(features, k, num_iters=100):
    N, D = features.shape
    assert N >= k, 'Number of clusters cannot be greater than number of points'
    index = np.random.choice(N, size=k, replace=False)
    centers = features[index]
    assignments = np.zeros(N, dtype=np.uint32)
    prev_assignments = None
    for n in range(num_iters):
        distances = cdist(features, centers)
        assignments = np.argmin(distances, axis=1)
        for j in range(k):
            points_in_cluster = features[assignments == j]
            if len(points_in_cluster) > 0:
                centers[j] = points_in_cluster.mean(axis=0)
        if (assignments == prev_assignments).all():
            break
        prev_assignments = assignments
    return assignments


def hierarchical_clustering(features, k):
    pass
    # """ Run the hierarchical agglomerative clustering algorithm.
    #
    # The algorithm is conceptually simple:
    #
    # Assign each point to its own cluster
    # While the number of clusters is greater than k:
    #     Compute the distance between all pairs of clusters
    #     Merge the pair of clusters that are closest to each other
    #
    # We will use Euclidean distance to define distance between clusters.
    #
    # Recomputing the centroids of all clusters and the distances between all
    # pairs of centroids at each step of the loop would be very slow. Thankfully
    # most of the distances and centroids remain the same in successive
    # iterations of the outer loop; therefore we can speed up the computation by
    # only recomputing the centroid and distances for the new merged cluster.
    #
    # Even with this trick, this algorithm will consume a lot of memory and run
    # very slowly when clustering large set of points. In practice, you probably
    # do not want to use this algorithm to cluster more than 10,000 points.
    #
    # Hints
    # - You may find pdist (imported from scipy.spatial.distance) useful
    #
    # Args:
    #     features - Array of N features vectors. Each row represents a feature
    #         vector.
    #     k - Number of clusters to form.
    #
    # Returns:
    #     assignments - Array representing cluster assignment of each point.
    #         (e.g. i-th point is assigned to cluster assignments[i])
    # """
    #
    #
    #
    # N, D = features.shape
    #
    # assert N >= k, 'Number of clusters cannot be greater than number of points'
    #
    # assignments = np.arange(N, dtype=np.uint32)
    # centers = np.copy(features)
    # n_clusters = N
    #
    #
    # while n_clusters > k:
    #     ### YOUR CODE HERE
    #     pass
    #     ### END YOUR CODE
    #
    # return assignments


def color_features(img):
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))
    features = img.reshape((H * W, C))
    return features


def color_position_features(img):
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))
    y, x = np.mgrid[:H,:W]
    features[:,:C] = color.reshape((H * W, C))
    features[:,C:] = np.dstack((x, y)).reshape((H * W, 2))
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean) / std
    return features


def compute_accuracy(mask_gt, mask):
    accuracy = None
    mask_comparison = mask_gt == mask
    accuracy = np.mean(mask_comparison)
    return accuracy


def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """
    num_segments = np.max(segments) + 1
    best_accuracy = 0
    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)
    return best_accuracy

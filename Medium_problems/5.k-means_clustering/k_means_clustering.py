from typing import List, Tuple

def calculate_centroid(points: List[Tuple[float, ...]]) -> Tuple[float, ...]:
    if not points:
        return tuple()
    dimension = len(points[0])
    n = len(points)
    centroid = [0.0] * dimension
    for point in points:
        for i in range(dimension):
            centroid[i] += point[i]
    return tuple(coord / n for coord in centroid)

def euclidean_distance(point1: Tuple[float, ...], point2: Tuple[float, ...]) -> float:
    return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5

def k_means_clustering(points: List[Tuple[float, ...]], k: int, initial_centroids: List[Tuple[float, ...]], max_iterations: int) -> List[Tuple[float, ...]]:
    if k <= 0 or len(initial_centroids) != k or len(points) < k:
        raise ValueError("Invalid number of clusters or initial centroids.")
    
    dimension = len(points[0])
    for point in points + initial_centroids:
        if len(point) != dimension:
            raise ValueError("All points and centroids must have the same dimension.")
    
    final_centroids = initial_centroids.copy()

    for _ in range(max_iterations):
        clusters = {i: [] for i in range(k)}
        for point in points:
            closest_centroid = min(range(k), key=lambda i: euclidean_distance(point, final_centroids[i]))
            clusters[closest_centroid].append(point)
        
        new_centroids = [
            calculate_centroid(clusters[i]) if clusters[i] else final_centroids[i]
            for i in range(k)
        ]

        if new_centroids == final_centroids:
            break

        final_centroids = new_centroids

    # Round each coordinate to 4 decimal places
    return [tuple(round(x, 4) for x in centroid) for centroid in final_centroids]

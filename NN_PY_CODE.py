import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Function to compute nearest neighbor distances and angles in degrees
def nearest_neighbor_analysis(points):
    n_points = len(points)
    dist_matrix = distance_matrix(points, points)

    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(dist_matrix, np.inf)

    # Find nearest neighbors
    nearest_neighbors = np.argmin(dist_matrix, axis=1)
    nearest_distances = np.min(dist_matrix, axis=1)

    # Calculate angles in degrees (polar coordinates)
    vectors = points[nearest_neighbors] - points
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))

    return nearest_distances, angles

# Function to estimate nearest neighbor orientation density in degrees
def orientation_density(angles, bandwidth, angle_range=(0, 360)):
    angle_bins = np.linspace(angle_range[0], angle_range[1], 360)
    density = np.zeros_like(angle_bins)

    for angle in angles:
        density += np.exp(-0.5 * ((angle_bins - angle) / bandwidth) ** 2)

    density /= (bandwidth * np.sqrt(2 * np.pi))
    return angle_bins, density

# Load data from CSV file
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    points = data[['X', 'Y']].values
    return points

# Parameters
bandwidth = 5  # Bandwidth in degrees

# File path to the CSV file
file_path = 'X_nodes_coordinates.csv'

# Load points from the provided CSV file
points = load_data_from_csv(file_path)

# Perform nearest neighbor analysis
nearest_distances, angles = nearest_neighbor_analysis(points)

# Estimate orientation density
angle_bins, density = orientation_density(angles, bandwidth)

# Plot results
plt.figure(figsize=(12, 6))

# Plot points and nearest neighbors
plt.subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], c='blue', label='Points')
for i, point in enumerate(points):
    nearest_point = points[np.argmin(distance_matrix([point], points))]
    plt.plot([point[0], nearest_point[0]], [point[1], nearest_point[1]], 'r--', linewidth=0.5)
plt.title('Nearest Neighbors')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

# Plot orientation density in degrees
plt.subplot(1, 2, 2)
plt.plot(angle_bins, density, color='red', lw=2)
plt.title('Orientation Density (Degrees)')
plt.xlabel('Angle (degrees)')
plt.ylabel('Density')

plt.show()

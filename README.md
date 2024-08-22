Nearest Neighbor Analysis Documentation 
-	Shubham J.
Overview
This code performs a nearest neighbor analysis on spatial point patterns to compute distances and angles between nearest neighbors. It then estimates and visualizes the orientation density of these angles.
Components
1.	Nearest Neighbor Analysis: Calculates the distance and angle between each point and its nearest neighbor.
2.	Orientation Density Estimation: Estimates the density of nearest neighbor angles using a Gaussian kernel.
3.	Data Loading: Imports spatial point data from a CSV file.
4.	Visualization: Plots the spatial distribution of points, their nearest neighbors, and the orientation density of the nearest neighbor angles.
Functions
1. nearest_neighbor_analysis(points)
Purpose: Computes the nearest neighbor distances and angles for a given set of points.
Parameters:
•	points (numpy.ndarray): A 2D array where each row represents the coordinates of a point (e.g., [[x1, y1], [x2, y2], ...]).
Returns:
•	nearest_distances (numpy.ndarray): Array of distances to the nearest neighbor for each point.
•	angles (numpy.ndarray): Array of angles (in degrees) between each point and its nearest neighbor.
Details:
•	Computes the distance matrix for all points.
•	Sets the diagonal of the distance matrix to infinity to exclude self-distances.
•	Finds the nearest neighbor for each point by selecting the minimum distance.
•	Calculates angles using the arctan2 function and converts them to degrees.
Code:
python
Copy code
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
2. orientation_density(angles, bandwidth, angle_range=(0, 360))
Purpose: Estimates the density of nearest neighbor angles using a Gaussian kernel.
Parameters:
•	angles (numpy.ndarray): Array of angles (in degrees) between points and their nearest neighbors.
•	bandwidth (float): Bandwidth for the Gaussian kernel in degrees.
•	angle_range (tuple of floats): Range of angles for which to estimate the density (default is (0, 360)).
Returns:
•	angle_bins (numpy.ndarray): Array of angle bin edges.
•	density (numpy.ndarray): Estimated density of angles for each bin.
Details:
•	Uses a Gaussian kernel to estimate the density function.
•	The kernel function smooths the angle distribution based on the specified bandwidth.
Code:
python
Copy code
def orientation_density(angles, bandwidth, angle_range=(0, 360)):
    angle_bins = np.linspace(angle_range[0], angle_range[1], 360)
    density = np.zeros_like(angle_bins)

    for angle in angles:
        density += np.exp(-0.5 * ((angle_bins - angle) / bandwidth) ** 2)

    density /= (bandwidth * np.sqrt(2 * np.pi))
    return angle_bins, density
3. load_data_from_csv(file_path)
Purpose: Loads spatial point data from a CSV file.
Parameters:
•	file_path (str): Path to the CSV file containing the point coordinates.
Returns:
•	points (numpy.ndarray): Array of points loaded from the CSV file.
Details:
•	Reads the CSV file and extracts the 'X' and 'Y' coordinates as a numpy array.
Code:
python
Copy code
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    points = data[['X', 'Y']].values
    return points
Usage
1.	Set Parameters:
o	Define the bandwidth for density estimation.
o	Specify the path to the CSV file containing the point coordinates.
2.	Load Data:
python
Copy code
file_path = 'X_nodes_coordinates.csv'
points = load_data_from_csv(file_path)
3.	Perform Analysis:
python
Copy code
nearest_distances, angles = nearest_neighbor_analysis(points)
4.	Estimate Orientation Density:
python
Copy code
bandwidth = 5  # Bandwidth in degrees
angle_bins, density = orientation_density(angles, bandwidth)
5.	Visualize Results:
python
Copy code
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
Example Output
The resulting plots will display:
•	A scatter plot of the spatial points and lines connecting each point to its nearest neighbor.
•	A plot of the orientation density function showing the distribution of angles between nearest neighbors.

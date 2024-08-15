import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

def trilaterate_gps(surrounding_positions, distances, error_radii, sat_pos, alpha=1):
    trajectory = []
    errors = []

    # Define the objective function for optimization
    def objective(x):
        est_pos = np.array(x)
        error = sum(((geopy.distance.geodesic(est_pos, np.array(pos)).meters - dist) / error_radius**alpha) ** 2
                   for pos, dist, error_radius in zip(surrounding_positions, distances, error_radii))
        trajectory.append(est_pos)  # Record the trajectory
        errors.append(error)  # Record the error
        return error

    # Initial guess: centroid of surrounding positions
    initial_guess = sat_pos
    # Define bounds for latitude and longitude
    lat_bounds = (min(pos[0] for pos in surrounding_positions) - 0.0001,
                  max(pos[0] for pos in surrounding_positions) + 0.0001)
    lon_bounds = (min(pos[1] for pos in surrounding_positions) - 0.0001,
                  max(pos[1] for pos in surrounding_positions) + 0.0001)
    bounds = [lat_bounds, lon_bounds]

    # Perform the minimization with method 'SLSQP' and tolerance settings
    result = minimize(objective, initial_guess, bounds=bounds)

    return result.x, trajectory, errors, objective

# Example usage:
surrounding_positions=[(34.937903803424504, 31.960983236281198), (34.93504752987018, 31.95739056177799), (34.935410256259196, 31.958836120605252), (34.93549895981924, 31.95966816181387), (34.93638701448896, 31.95786585704581), (34.9354375184986, 31.960305451179785), (34.93531479785182, 31.958666802649784), (34.93524672328245, 31.95838290093478)]
distances =[297.0556793063685, 283.5255221648918, 147.728662692188, 74.0512911761465, 260.604267292448, 28.816720263416553, 175.61870578415687, 202.2950699184737]
error_radii=[0.1, 4.355329973781741, 5.698153582081583, 9.310724247246107, 9.514797075298988, 16.195485645533918, 16.506614149065815, 18.183443567368776]
real_Pos=(34.93525731808676, 31.960488681993382)

sat_pos = np.mean(surrounding_positions, axis=0)
estimated_position, trajectory, errors, objective = trilaterate_gps(surrounding_positions, distances, error_radii, sat_pos)

# Convert trajectory and errors to numpy arrays for easier plotting
trajectory = np.array(trajectory)
errors = np.array(errors)
print(trajectory.shape)
trajectory=trajectory[0:300:10]
errors=errors[0:300:10]
print(trajectory.shape)
# Determine the bounds based on the surrounding positions
lat_min = min(pos[0] for pos in surrounding_positions) - 0.0001
lat_max = max(pos[0] for pos in surrounding_positions) + 0.0001
lon_min = min(pos[1] for pos in surrounding_positions) - 0.0001
lon_max = max(pos[1] for pos in surrounding_positions) + 0.0001

# Create a grid of latitude and longitude values within the bounds
lat_values = np.linspace(lat_min, lat_max, 100)
lon_values = np.linspace(lon_min, lon_max, 100)
lat_grid, lon_grid = np.meshgrid(lat_values, lon_values)

# Calculate the error for each point in the grid
error_grid = np.zeros_like(lat_grid)
for i in range(lat_grid.shape[0]):
    for j in range(lat_grid.shape[1]):
        error_grid[i, j] = objective([lat_grid[i, j], lon_grid[i, j]])
# Plotting the error function and the spatial convergence trajectory
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(lat_grid, lon_grid, error_grid, cmap='viridis', alpha=0.6)
ax.scatter([pos[0] for pos in surrounding_positions], [pos[1] for pos in surrounding_positions], color='blue', label='Surrounding Positions')
ax.scatter(sat_pos[0], sat_pos[1], errors[0], color='yellow', label='Initial Guess')
ax.scatter(estimated_position[0], estimated_position[1], errors[-1],s=30,color='green', label='Estimated Position')
ax.scatter(real_Pos[0], real_Pos[1], errors[-1], color='purple',s=30, label='Real Position')

# Plot the trajectory from the initial guess to the estimated position with smaller red dots
ax.plot(trajectory[:, 0], trajectory[:, 1], errors, color='red', linestyle='-', label='Optimization Path')
ax.scatter(trajectory[:, 0], trajectory[:, 1], errors, color='red', s=10, label='Optimization Path Points')  # Smaller red dots

ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Error (Sum of Squares)')
ax.set_title('Spatial Convergence Trajectory with Reduced Domain Error Landscape')
ax.legend()
plt.show()

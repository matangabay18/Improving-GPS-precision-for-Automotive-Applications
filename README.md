
# Enhanced GPS for Mobile Car Applications

This project aims to improve the accuracy of GPS positioning for mobile car applications by integrating satellite GPS data with additional error correction techniques using surrounding vehicle and RSU data.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Classes and Functions](#classes-and-functions)
6. [Experiments](#experiments)
7. [Simulation Parameters](#simulation-parameters)
8. [Proof of Concept: Inertial Navigation Integration](#proof-of-concept-inertial-navigation-integration)

## Introduction

The project leverages the `traci` library for SUMO (Simulation of Urban MObility) and the `geopy` library for geodesic calculations to simulate and improve GPS accuracy for vehicular networks. The solution incorporates satellite GPS data with error correction based on nearby vehicles and Road Side Units (RSUs).

## Features

- **Simulated GPS Error Addition**: Adds realistic errors to GPS coordinates to simulate real-world scenarios.
- **Error Correction**: Uses surrounding vehicle data and RSUs to correct GPS errors using trilateration.
- **Performance Metrics**: Computes Mean Squared Error (MSE) and absolute errors to evaluate the performance.
- **Simulation Configurations**: Supports different simulation environments like urban settings and highways.

## Installation

1. **Ensure SUMO is installed and configured on your system.**

2. **Download the necessary directories:**
    - `test_Line`
    - `Loud_City_NY`
    - `High_way`

3. **Change the path to the simulation configuration files (`osm.sumocfg`) depending on where they are saved on your computer.**

## Usage

1. **Run the simulation:**
    ```bash
    python run_simulation.py
    ```

2. **Select the simulation environment:**
    - `1` for Test Line
    - `2` for Loud City NY
    - `3` for Highway

3. **View the results printed in the console, including estimated positions, errors, and performance metrics.

## Classes and Functions

### `VehicleTracker`

Tracks vehicles' GPS data, adds errors, and corrects them using nearby vehicles and RSUs.

- `update_vehicle_data(vehicle_id, geo_position, speed, step)`: Updates vehicle data with GPS positions, speeds, and error.
- `find_nearby_vehicles_and_check_rsus(vehicle_ids, step)`: Finds nearby vehicles and checks RSU proximity.
- `print_estimated_positions_and_errors(alpha, want_print, better_flag)`: Prints estimated positions and calculates errors.
- `calculate_mse_satellite_positions()`: Calculates the MSE for satellite positions.
- `print_error_results()`: Prints error results.
- `sweep_alpha(vehicle_tracker, start, end, step, want_print, better_flag)`: Sweeps different alpha values for optimization.
- `sweep_neighbors(vehicle_tracker, start, end, step, want_print, better_flag)`: Sweeps different neighbor counts.

### `RSUManager`

Manages RSU locations and updates vehicle proximity.

- `update_vehicle_proximity(vehicle_id, vehicle_geo_position, step)`: Updates proximity information for a vehicle.
- `print_vehicles_near_rsus()`: Prints vehicles near each RSU.

### `generate_rsu_grid(point1, point2, point3, point4, interval_km)`

Generates a grid of RSUs within defined boundaries.

### `run_simulation()`

Main function to run the simulation, update vehicle data, and print results.

## Experiments

The project includes several experiments that can be conducted:

1. **Neighbors Sweep:** Evaluate performance with different numbers of neighbors.
    ```python
    vehicle_tracker.sweep_neighbors(vehicle_tracker, start=lowest_num_neighbors_use, end=highest_num_neighbors_in_use, step=jumps, want_print=True, better_flag=True)
    ```

2. **Alpha Sweep:** Evaluate performance with different alpha values for the optimization function.
    ```python
    vehicle_tracker.sweep_alpha(vehicle_tracker, start=initial_value, end=final_value, step=jumps, want_print=True, better_flag=True)
    ```

## Simulation Parameters

The following parameters can be adjusted to customize the simulation:

- **use_RSU**: Boolean to enable or disable the use of RSUs in the simulation.
- **better_flag**: Boolean to choose whether to use the improved method.
- **proximity_radius**: Radius in meters to consider for nearby vehicles and RSUs.
- **error_std_dev**: Standard deviation of the error to add to the GPS coordinates (in meters).
- **number_of_steps**: Number of simulation steps to run.
- **specific_car_id**: ID of the specific car to track in the simulation.
- **num_of_neighbors**: Number of neighboring vehicles to consider for error correction.
- **interval_km**: Determines the distance between each RSU in the grid that is produced.

## Proof of Concept: Inertial Navigation Integration

A proof of concept (POC) for integrating inertial navigation into the existing GPS system has been developed. This integration leverages inertial navigation data to balance and enhance the accuracy of the GPS position estimations through a weighted combination of inertial navigation and trilateration results.

### Key Features of Inertial Navigation Integration:

- **Position Estimation**: Estimates the next vehicle position based on current position, speed, heading, time step, and acceleration.
- **Weighted Position Calculation**: Combines inertial navigation and trilateration positions using a weighted approach.
- **Enhanced Accuracy**: Provides more accurate position estimations by reducing error distances through the integration of multiple data sources.

### Example Usage

In the simulation loop, the estimated position from inertial navigation is calculated, and then a weighted combination with the trilateration-based position is performed to achieve better accuracy.

```python
# Example of integrating inertial navigation with trilateration in the simulation loop

# Calculate the weighted position
weight_inertial = 0.5
weight_trilateration = 0.5
inertial_position = self.inertial_positions[actual_position_index-1]  # Use the inertial position
weighted_position = calculate_weighted_position(inertial_position, estimated_position, weight_inertial, weight_trilateration)

# Calculate the weighted error
weighted_error = calculate_distance(weighted_position, actual_position)
```

This integration demonstrates how combining GPS with inertial navigation can significantly enhance the accuracy and reliability of vehicle positioning systems in dynamic environments.
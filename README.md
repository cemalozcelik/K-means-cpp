# KMeans++ Clustering Algorithm

This project implements the KMeans++ clustering algorithm in C++. It includes classes for managing data points, performing clustering, and applying the KMeans++ algorithm.

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Description

The KMeans++ algorithm is a variation of the KMeans clustering algorithm that aims to improve the initial selection of cluster centroids. It selects the initial centroids in a way that distributes them more effectively across the data points.

This implementation consists of three main classes:
1. `Node`: Represents a data point with x and y coordinates.
2. `Clustering`: Manages the data points and provides functionality to load them from a text file.
3. `KMeansPP`: Applies the KMeans++ algorithm using the provided data points and centroids.

## Getting Started

### Prerequisites

- C++ compiler (e.g., g++ for Unix-based systems)
- Visual Studio Code (recommended, for a better coding experience)
- C/C++ extension for Visual Studio Code

### Installation

1. Clone the repository or download the source files.
2. Open the project folder in Visual Studio Code.
3. Install the C/C++ extension if not already installed.
4. Build and run the code as described in the [Usage](#usage) section.

## Usage

1. Adjust the data points in the `data.txt` file or replace it with your data.
2. Open the terminal in Visual Studio Code or any command-line interface.
3. Navigate to the project folder.
4. Compile the code by running:
```
g++ -o kmeans main.cpp `pkg-config --cflags --libs opencv4`
```
5. Run the executable:
```
./kmeans
```
6. Follow the prompts to enter the number of clusters and iterations.

# Image Segmentation using KMeans Clustering

This project performs image segmentation using the KMeans clustering algorithm implemented from scratch. The goal is to segment an image into distinct regions based on pixel color similarity.

## Table of Contents
- [Image Segmentation using KMeans Clustering](#image-segmentation-using-kmeans-clustering)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [KMeans Clustering](#kmeans-clustering)
  - [Image Segmentation](#image-segmentation)
  - [Installation](#installation)
  - [Usage](#usage)

## Introduction
Image segmentation is a process of partitioning an image into multiple segments or regions. The purpose of segmentation is to simplify and/or change the representation of an image into something more meaningful and easier to analyze.

## KMeans Clustering
KMeans clustering is an unsupervised machine learning algorithm that partitions a dataset into K clusters, where each data point belongs to the cluster with the nearest mean. The algorithm follows these steps:
1. Initialize K centroids randomly.
2. Assign each data point to the nearest centroid, forming K clusters.
3. Recalculate the centroids as the mean of all points in each cluster.
4. Repeat steps 2 and 3 until convergence or a maximum number of iterations.

## Image Segmentation
In this project, KMeans clustering is used to segment an image by grouping similar pixels based on their color values. Each pixel is treated as a data point in a 3-dimensional space (RGB values).

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/aymen-000/image_segmentation.git
    cd image_segmentation
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To run the image segmentation script, use the following command:
```sh
python main.py path/to/your/image.jpg path/to/save/original.jpg path/to/save/segmented.jpg --k 3 --max_iters 100
```
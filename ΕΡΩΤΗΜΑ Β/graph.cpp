#include <iostream>
#include <algorithm> // Include for sorting and other algorithms
#include <random> // Include for random number generation
#include "graph.h"
#include "global_functions.h"

// Constructor for the Graph class, initializes the graph with a given size
Graph::Graph(int size) : nodes(size) {}

// Function to add an edge between two nodes in the graph
void Graph::addEdge(int src, int dest) {
    nodes[src].neighbors.insert(dest);
}

// Function to get the neighbors of a given node in the graph
const std::set<int>& Graph::getNeighbors(int nodeIndex) const {
    return nodes[nodeIndex].neighbors;
}

// Function to build a k-Nearest Neighbors Graph using LSH
Graph buildKNNG(LSH &lsh, int k, int datasetSize) {
    Graph kNNG(datasetSize);

    for (int i = 0; i < datasetSize; ++i) {
        const std::vector<unsigned char>& queryPoint = lsh.getDataset()[i];

        kNNG.storePoint(queryPoint);

        // Query for the k nearest neighbors of the point
        auto neighbors = lsh.queryNNearestNeighbors(queryPoint, k);
        for (const auto& neighbor : neighbors) {
            int neighborIndex = neighbor.first;
            kNNG.addEdge(i, neighborIndex); // Add edges to the graph
        }
    }
    return kNNG;
}

// Function to build a k-Nearest Neighbors Graph using Hypercube method
Graph buildKNNG_H(Hypercube &hypercube, int k, int datasetSize) {
    Graph kNNG(datasetSize);

    for (int i = 0; i < datasetSize; ++i) {
        const std::vector<unsigned char>& queryPoint = hypercube.getDataset()[i];

        kNNG.storePoint(queryPoint);

        // Query for the k nearest neighbors of the point
        auto neighbors = hypercube.kNearestNeighbors(queryPoint, k);
        for (const auto& neighbor : neighbors) {
            int neighborIndex = neighbor.first;
            kNNG.addEdge(i, neighborIndex); // Add edges to the graph
        }
    }

    return kNNG;
}

// Function to store a point in the graph
void Graph::storePoint(const std::vector<unsigned char>& point) {
    dataPoints.push_back(point);
}

// Function to get a point from the graph using its node index
const std::vector<unsigned char>& Graph::getPoint(int nodeIndex) const {
    return dataPoints[nodeIndex];
}

// Function to get the size of the graph
std::size_t Graph::size() const {
    return nodes.size();
}

// Greedy Nearest Neighbor Search (GNNS) function
std::vector<std::pair<int, double>> Graph::GNNS(const std::vector<unsigned char>& queryPoint, int N, int R, int T, int E) const {
    std::vector<std::pair<int, double>> potentialNeighbors;

    std::random_device rd;
    std::default_random_engine engine(rd());

    for (int r = 0; r < R; ++r) {
        std::uniform_int_distribution<int> distribution(0, this->size() - 1);
        int currentNode = distribution(engine);

        for (int t = 0; t < T; ++t) {
            const auto& neighbors = this->getNeighbors(currentNode);
            int bestNeighbor = currentNode;
            double bestDistance = euclideanDistance(this->getPoint(currentNode), queryPoint);
            bool isLocalOptimal = true;

            int count = 0;
            for (int neighbor : neighbors) {
                if (count >= E) break;
                double distance = euclideanDistance(this->getPoint(neighbor), queryPoint);
                potentialNeighbors.emplace_back(neighbor, distance); // Store each neighbor along with the distance

                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestNeighbor = neighbor;
                    isLocalOptimal = false;
                }
                count++;
            }

            // Terminate early if current node is better than its neighbors (local optimal)
            if (isLocalOptimal) {
                break;
            }

            currentNode = bestNeighbor;
        }
    }
    // Sort potential neighbors based on their distance to the query point
    std::sort(potentialNeighbors.begin(), potentialNeighbors.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second < b.second;
              });

    // Remove duplicate entries from the list of potential neighbors
    auto last = std::unique(potentialNeighbors.begin(), potentialNeighbors.end(),
                            [](const std::pair<int, double>& a, const std::pair<int, double>& b){
                                return a.first == b.first;
                            });
    potentialNeighbors.erase(last, potentialNeighbors.end());

    // Keep only the top N elements in the list of potential neighbors
    if (potentialNeighbors.size() > N) {
        potentialNeighbors.resize(N);
    }
    return potentialNeighbors;
}

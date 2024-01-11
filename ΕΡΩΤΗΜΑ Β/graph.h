//
// Created by test on 11/20/2023.
//

#ifndef PROJECT_K23_SEC_GRAPH_H
#define PROJECT_K23_SEC_GRAPH_H

#include <vector>
#include <set>
#include "lsh_class.h" // Include your LSH class header
#include "Hypercube.h" // Include your Hypercube class header

// Define a Node for the Graph
struct Node {
    std::set<int> neighbors; // Set of indices of the neighbors
};

// Define the Graph class
class Graph {
public:
    explicit Graph(int size);
    void addEdge(int src, int dest);
    [[nodiscard]] const std::set<int>& getNeighbors(int nodeIndex) const;
    [[nodiscard]] std::size_t size() const; // Returns the number of nodes in the graph
    [[nodiscard]] const std::vector<unsigned char>& getPoint(int nodeIndex) const; // Returns the data point for a given node
    // Method to store a data point
    void storePoint(const std::vector<unsigned char>& point);
    [[nodiscard]] std::vector<std::pair<int, double>> GNNS(const std::vector<unsigned char>& queryPoint, int K, int R, int T, int E) const;
private:
    std::vector<Node> nodes;
    std::vector<std::vector<unsigned char>> dataPoints;

};

// Function to construct the k-NNG using LSH
Graph buildKNNG(LSH &lsh, int k, int datasetSize);
// Function to construct the k-NNG using Hypercube
Graph buildKNNG_H(Hypercube &hypercube, int k, int datasetSize);



#endif //PROJECT_K23_SEC_GRAPH_H

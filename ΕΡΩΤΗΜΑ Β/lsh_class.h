#ifndef LSH_H
#define LSH_H

#include <vector>
#include <random>
#include "mnist.h"

class LSH {
public:
    explicit LSH(std::vector<std::vector<unsigned char>> dataset,int num_dimensions, int k = 4, int L = 5, int N = 1, double R = 10000);
    ~LSH();

    // Function to create hash functions
    [[nodiscard]] std::vector<std::pair<std::vector<double>, double>> createHashFunctions(int nf, int nd) const;

    // Overload 1: Doesn't take radius, uses the class's private member R
    std::vector<int> rangeSearch(const std::vector<unsigned char>& query_point);

    // Overload 2: Takes a radius and uses that
    std::vector<int> rangeSearch(const std::vector<unsigned char>& query_point, double radius);

    // Function to query N nearest neighbors for a given query point
    std::vector<std::pair<int, double>> queryNNearestNeighbors(const std::vector<unsigned char>& query_point,int K);

    // Getter for N
    [[nodiscard]] int returnN() const;
    [[nodiscard]] double returnR() const;

    // Function to get the dataset
    const std::vector<std::vector<unsigned char>>& getDataset() const;

    // Function to print the hash tables
    void printHashTables();

private:
    int k; // Number of hash functions
    int L; // Number of hash tables
    int num_buckets = 15000; // Number of buckets
    int N; // Number of nearest neighbors
    double w; // Bucket width
    double R; // Radius

    std::vector<std::vector<unsigned char>> dataset; // The dataset of points
    std::vector<int> ri_values; // Random coefficients

    //int num_dimensions = 784; // Number of dimensions of a data point
    int num_dimensions;
    // to store both the index and the ID value.
    std::vector<std::vector<std::vector<std::pair<int, int>>>> hash_tables;

    // Hash functions for each table
    std::vector<std::vector<std::pair<std::vector<double>, double>>> hash_functions;

    // Helper function to build the hash table index
    void buildIndex();

    // Helper function to compute the ID value for a data point
    int64_t computeID(const std::vector<unsigned char>& data_point, int table_index);
};

#endif
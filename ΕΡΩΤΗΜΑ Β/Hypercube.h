#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include <vector>
#include <random>
#include <set>

class Hypercube {
public:
    explicit Hypercube(std::vector<std::vector<unsigned char>> dataset,
              int k = 14,int M=6000,int probes=10, int N = 1, double R = 10000);
    ~Hypercube();


    std::vector<std::pair<int, double>> kNearestNeighbors(const std::vector<unsigned char>& q,int K);
    std::vector<int> rangeSearch(const std::vector<unsigned char>& q);
    std::vector<int> rangeSearch(const std::vector<unsigned char>& q, double radius);

    // Function to get the dataset
    const std::vector<std::vector<unsigned char>>& getDataset() const;

    [[nodiscard]] int returnN() const;
    [[nodiscard]] double returnR() const;

private:
    // Member variables
    std::vector<std::vector<unsigned char>> dataset;
    int k;
    int num_dimensions = 784;
    int N;
    double R;
    double w;
    int reduced_dimension;
    int M;
    int n=60000;
    int probes;
    std::vector<std::vector<float>> random_projection_matrix;
    std::vector<std::vector<int>> hash_table;
    std::vector<std::pair<std::vector<float>, float>> table_functions;
    std::mt19937 generator;

    void buildIndex();

    // Computes the ID value for a data point
    int computeID(const std::vector<unsigned char>& data_point);

    // Defines the function to map hi values to {0, 1}
    int fi(int hi_value);

    // Returns candidates by probing the hypercube for the given query point
    std::vector<int> probe(const std::vector<unsigned char>& query_point, int maxHammingDistance);
    // Creates hash functions for the hypercube
    std::vector<std::pair<std::vector<float>, float>> createHashFunctions(int k, int dim);

    // Reduces the dimensionality of the data point using the random projection matrix
    std::vector<float> reduceDimensionality(const std::vector<unsigned char>& data_point);



};

#endif // HYPERCUBE_H

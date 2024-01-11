#include "Hypercube.h"
#include <iostream>
#include <random>
#include <cmath>
#include <set>
#include <algorithm>
#include <queue>
#include "global_functions.h"  // Make sure this contains the computeDPrime function

Hypercube::Hypercube(std::vector<std::vector<unsigned char>> dataset,
                     int k,int M,int probes,
                     int N, double R)
        : dataset(std::move(dataset)),
          k(k),
          M(M),probes(probes),
          N(N), R(R),
          table_functions(createHashFunctions(k, computeDPrime(n)))
{
    generator = std::mt19937(std::random_device{}());
    reduced_dimension = computeDPrime(n);
    //std::cout << "Reduced dimension: " << reduced_dimension << std::endl;
    // Resize the hash_table for 2^k buckets, each initialized with an empty vector
    hash_table.resize(1 << k);

    // Create the random projection matrix
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < reduced_dimension; ++i) {
        std::vector<float> v;
        for (int j = 0; j < num_dimensions; ++j) {
            v.push_back(distribution(generator));
        }
        random_projection_matrix.push_back(v);
    }

    std::uniform_real_distribution<double> w_distribution(400, 500);
    w = w_distribution(generator);

    buildIndex();

}

Hypercube::~Hypercube() {
    table_functions.clear();
}

void Hypercube::buildIndex() {
    for (int i = 0; i < dataset.size(); ++i) {
        int id = computeID(dataset[i]);
        hash_table[id].push_back(i);
    }
}

std::vector<int> Hypercube::probe(const std::vector<unsigned char>& query_point, int maxProbes) {
    //std::cout << "Hi" << std::endl;
    int hash_value = computeID(query_point);
    //std::cout << "hash_value: " << hash_value << std::endl;
    std::set<int> candidates;
    int vertices_checked = 0;
    int distance = 0;

    //std::cout << "maxProbes: " << maxProbes << std::endl;
    //std::cout << "vertices_checked: " << vertices_checked << std::endl;

    while(vertices_checked < maxProbes) {
        for (int i = 0; i < k && vertices_checked < maxProbes; ++i) {
            int neighbor_hash = hash_value ^ (1 << i);
            //std::cout << "Hello" << std::endl;
            // print hash_table[neighbor_hash]
            //std::cout << "hash_table[neighbor_hash].size(): " << hash_table[neighbor_hash].size() << std::endl;
            for(int idx : hash_table[neighbor_hash]) {
                //std::cout << "idx: " << idx << std::endl;
                candidates.insert(idx);
            }
            vertices_checked++;
        }
        distance++; // This increases the Hamming distance, but remember, you might want to constrain this if it goes beyond the bounds of your hypercube.
    }

    return {candidates.begin(), candidates.end()};
}


int Hypercube::fi(int hi_value) {
    //std::cout << "hi_value: " << hi_value << std::endl;
    //std::cout << "hi_value % 2: " << hi_value % 2 << std::endl;
    return hi_value % 2;
}

int Hypercube::computeID(const std::vector<unsigned char>& data_point) {

    // << "Hey!" << std::endl;
    auto reduced_data_point = reduceDimensionality(data_point);
    //std::cout << "Hey!!!" << std::endl;

    if (reduced_data_point.size() != reduced_dimension) {
        throw std::runtime_error("Reduced data point dimensions mismatch.");
    }

    int g_value = 0;
    for (int i = 0; i < k; ++i) {
        auto& [v, t] = table_functions[i];
        double dot_product = 0.0;
        for (int j = 0; j < reduced_dimension; ++j) {
            dot_product += v[j] * reduced_data_point[j];
        }
        int hi = static_cast<int>(std::floor((dot_product + t) / w));
        hi += 100000;
        g_value |= (fi(hi) << i);
    }
    return g_value;

}

std::vector<std::pair<std::vector<float>, float>> Hypercube::createHashFunctions(int k_, int dim) {
    std::vector<std::pair<std::vector<float>, float>> table_functions_;
    std::normal_distribution<float> distribution(0.0, 1.0);
    std::uniform_real_distribution<float> offset_distribution(0.0, w);

    for (int i = 0; i < k_; ++i) {
        std::vector<float> v;
        for (int j = 0; j < dim; ++j) {
            v.push_back(distribution(generator));
        }
        float t = offset_distribution(generator);
        table_functions_.emplace_back(v, t);
    }

    return table_functions_;
}

std::vector<std::pair<int, double>> Hypercube::kNearestNeighbors(const std::vector<unsigned char>& q,int K) {
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> nearest_neighbors_queue;
    std::vector<int> candidateIndices = probe(q, probes); // IT WAS k not probes


    int checkedCandidates = 0;
    for (const auto& index : candidateIndices) {
        if (checkedCandidates >= M) {
            break;
        }
        double distance = euclideanDistance(dataset[index], q);
        nearest_neighbors_queue.emplace(distance, index);
        checkedCandidates++;
    }

    std::vector<std::pair<int, double>> nearest_neighbors;

    int count = 0;
    while (!nearest_neighbors_queue.empty() && count < K) {
        nearest_neighbors.emplace_back(nearest_neighbors_queue.top().second, nearest_neighbors_queue.top().first);
        nearest_neighbors_queue.pop();
        count++;
    }

    return nearest_neighbors;
}


// Overload 1: Doesn't take a radius, uses the class's private member R
std::vector<int> Hypercube::rangeSearch(const std::vector<unsigned char>& q) {
    return rangeSearch(q, R);
}

// Overload 2: Takes a radius and uses that
std::vector<int> Hypercube::rangeSearch(const std::vector<unsigned char>& q, double radius) {

    std::vector<int> candidateIndices = probe(q, probes); //
    // print candiateIndices
    //std::cout << candidateIndices.size() << std::endl;

    std::set<int> inRangeIndices; // Use a set to avoid duplicates
    int checkedCandidates = 0;
    //std::cout << "candidateIndices.size(): " << candidateIndices.size() << std::endl;

    //std::cout << "Radius inside rangeSearch" << radius << std::endl;
    for (const auto& index : candidateIndices) {
        if (checkedCandidates >= M) {
            break;
        }
        double distance = euclideanDistance(dataset[index], q);
        //std::cout << "distance: " << distance << std::endl;
        //std::cout << "radius: " << radius << std::endl;
        if (distance <= radius) { // Use the provided radius
            inRangeIndices.insert(index);

        }
        checkedCandidates++;
    }

    // Convert the set to a vector for the final result
    std::vector<int> result(inRangeIndices.begin(), inRangeIndices.end());


    return result;
}


std::vector<float> Hypercube::reduceDimensionality(const std::vector<unsigned char>& data_point) {
    //std::cout << "Entering reduceDimensionality function." << std::endl;
    //std::cout << "data_point.size(): " << data_point.size() << std::endl;
    std::vector<float> reduced_point(reduced_dimension, 0.0);
    for (int i = 0; i < reduced_dimension; ++i) {
        for (int j = 0; j < num_dimensions; ++j) {
            reduced_point[i] += static_cast<float>(data_point[j]) * random_projection_matrix[i][j];
            //std::cout << reduced_point[i] << std::endl;
        }
    }

    //std::cout << "Exiting reduceDimensionality function." << std::endl;
    //std::cout << "reduced_point.size(): " << reduced_point.size() << std::endl;
    return reduced_point;
}

const std::vector<std::vector<unsigned char>>& Hypercube::getDataset() const {
    return dataset;
}

int Hypercube::returnN() const {
    return N;
}

double Hypercube::returnR() const {
    return R;
}

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <set>
#include <algorithm>

#include <queue>
#include "lsh_class.h"
#include "global_functions.h"


// LSH Constructor
LSH::LSH(std::vector<std::vector<unsigned char>> dataset,int num_dimensions, int k, int L, int N, double R)
        : dataset(std::move(dataset)),
          num_dimensions(num_dimensions),
          k(k), L(L),
          N(N), R(R),
          hash_tables(L, std::vector<std::vector<std::pair<int, int>>>(num_buckets)),
          hash_functions(L)
{

    // Δημιουργία των hash functions για κάθε table
    for(int i = 0; i < L; ++i) {
        hash_functions[i] = createHashFunctions(k, num_dimensions);
    }

    // Δημιουργία τυχαίων τιμών 'ri' για τα hash functions
    ri_values.resize(k);
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max()); // Range for int
    for (int i = 0; i < k; ++i) {
        ri_values[i] = dist(generator);
    }

    // Generate 'w' randomly in the range [400, 500]

    std::uniform_real_distribution<double> w_distribution(400, 500);
    w = w_distribution(generator);
    buildIndex();
}

// LSH Destructor
LSH::~LSH() {
    for (auto& table_functions : hash_functions) {
        for (auto& hash_function : table_functions) {
            hash_function.first.clear(); // Clear the vector within each pair
        }
        table_functions.clear(); // Clear the table's vector of hash functions
    }
    hash_functions.clear(); // Clear the vector of hash function tables
}

void LSH::buildIndex() {
    // Resize the hash tables to have num_buckets buckets
    for (auto& table : hash_tables) {
        table.resize(num_buckets);
    }

    for (int i = 0; i < dataset.size(); ++i) {
        for (int table_index = 0; table_index < L; ++table_index) {
            int64_t id_value = computeID(dataset[i], table_index);
            //std::cout << "id_value: " << id_value << std::endl;
            int64_t hash_value = id_value % num_buckets; // id_value mod TableSize
            //std::cout << "hash_value: " << hash_value << std::endl;
            hash_tables[table_index][hash_value].emplace_back(i, id_value);
        }
    }
}


// Δημιουργία μιας λίστας από nf hash_functions για το LSH χρησιμοποιώντας κανονικές και ομοιόμορφες κατανομές
std::vector<std::pair<std::vector<double>, double>> LSH::createHashFunctions(int nf, int nd) const {
    // Αρχικοποίηση random number generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);
    std::uniform_real_distribution<double> uniform_dist(0, w);



    std::vector<std::pair<std::vector<double>, double>> local_hash_functions;
    local_hash_functions.reserve(nf);
    // Δημιουργήσαμε nf hash_functions που παράγουν και αποθηκεύουν v μεγέθους nd και t στο [0, w)
    for (int i = 0; i < nf; ++i) {
        std::vector<double> v(nd);
        for (int j = 0; j < nd; ++j) {
            v[j] = distribution(generator);
        }
        double t = uniform_dist(generator);
        local_hash_functions.emplace_back(v, t);
    }

    return local_hash_functions;
}

int64_t LSH::computeID(const std::vector<unsigned char>& data_point, int table_index) {
    if (table_index < 0 || table_index >= L) {
        throw std::out_of_range("Invalid table_index");
    }
    if (data_point.size() != num_dimensions) {
        throw std::invalid_argument("Invalid data_point dimensions");
    }

    auto& table_functions = hash_functions[table_index];

    if (table_functions.size() != k) {
        throw std::runtime_error("Invalid number of hash functions for the table.");
    }

    //const int64_t M = (1LL << 32) - 5; // This simply wont work, id_value!= query_id_value always with this M
    //const int64_t M = 1000000003; // Define M as a large prime
    const int64_t M = (1LL << 30) - 5;
    uint64_t id_value = 0;

    for (int i = 0; i < k; ++i) {
        auto& [v, t] = table_functions[i];
        double dot_product = 0.0;
        for (int j = 0; j < num_dimensions; ++j) {
            dot_product += v[j] * data_point[j];
        }
        int hi = static_cast<int>(std::floor((dot_product + t) / w));
        hi += 100000; // Ensure it's positive
        uint64_t ri_hi_mod_M = (ri_values[i] * hi) % M;
        id_value = (id_value + ri_hi_mod_M) % M;
        //std::cout << "id_value: " << id_value << std::endl;
    }
    // id_value = [(r1h1(p) + r2h2(p) + · · · + rkhk (p)) mod M]
    return id_value;
}

// Create function to print hash tables
void LSH::printHashTables() {
    for (int table_index = 0; table_index < L; ++table_index) {
        std::cout << "Table " << table_index << ":" << std::endl;
        for (int bucket_index = 0; bucket_index < num_buckets; ++bucket_index) {
            std::cout << "Bucket " << bucket_index << ": ";
            for (const auto& [index, id_value] : hash_tables[table_index][bucket_index]) {
                std::cout << index << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}


std::vector<std::pair<int, double>> LSH::queryNNearestNeighbors(const std::vector<unsigned char>& query_point, int K) {
    std::priority_queue<std::pair<double, int>> nearest_neighbors_queue;
    for (int table_index = 0; table_index < L; ++table_index) {
        int64_t query_id_value = computeID(query_point, table_index); // Compute the ID for the query_point
        int64_t hash_value = query_id_value % num_buckets;

        for (const auto& [candidate_index, id_value] : hash_tables[table_index][hash_value]) {
            // Only compute the distance if the ID of the data point matches the ID of the query_point
            if (id_value == query_id_value) {
                //std::cout << "id_value: " << id_value << std::endl;
                double distance = euclideanDistance(dataset[candidate_index], query_point);
                nearest_neighbors_queue.emplace(distance, candidate_index);
            }
        }
    }

    std::vector<std::pair<int, double>> nearest_neighbors;
    while (!nearest_neighbors_queue.empty() && nearest_neighbors.size() < K) {
        nearest_neighbors.emplace_back(nearest_neighbors_queue.top().second, nearest_neighbors_queue.top().first);
        nearest_neighbors_queue.pop();
    }

    std::reverse(nearest_neighbors.begin(), nearest_neighbors.end());

    return nearest_neighbors;
}

// Overload 1: Doesn't take radius, uses the class's private member R
std::vector<int> LSH::rangeSearch(const std::vector<unsigned char>& query_point) {
    return rangeSearch(query_point, R); // Call the second overload using the class's private member R
}

// Overload 2: Takes a radius and uses that
std::vector<int> LSH::rangeSearch(const std::vector<unsigned char>& query_point, double radius) {
    std::set<int> candidates_within_radius;

    //std::cout << "radius: " << radius << std::endl;


    for (int table_index = 0; table_index < L; ++table_index) {
        int query_id_value = computeID(query_point, table_index);
        int hash_value = query_id_value % num_buckets;

        for (const auto& [candidate_index, id_value] : hash_tables[table_index][hash_value]) {

            if (id_value == query_id_value) {
                double distance = euclideanDistance(dataset[candidate_index], query_point);

                if (distance <= radius) {  // Use the passed radius
                    candidates_within_radius.insert(candidate_index);
                }
            }
        }
    }

    std::vector<int> result(candidates_within_radius.begin(), candidates_within_radius.end());
    return result;
}

const std::vector<std::vector<unsigned char>>& LSH::getDataset() const {
    return dataset;
}

int LSH::returnN() const {
    return N;
}

double LSH::returnR() const {
    return R;
}
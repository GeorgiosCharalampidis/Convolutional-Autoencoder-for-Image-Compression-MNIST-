//
// Created by test on 10/20/2023.
//

#ifndef PROJECTEM_GLOBAL_FUNCTIONS_H
#define PROJECTEM_GLOBAL_FUNCTIONS_H

#include <vector>

#include <stdexcept>


double euclideanDistance(const std::vector<unsigned char>& dataset, const std::vector<unsigned char>& query_set);
std::vector<unsigned char> convertToUnsignedChar(const std::vector<double>& vec);


int computeDPrime(int n);
std::vector<std::pair<int, double>> trueNNearestNeighbors(const std::vector<std::vector<unsigned char>>& dataset,
                                                          const std::vector<unsigned char>& query_point, int N);


#endif //PROJECTEM_GLOBAL_FUNCTIONS_H

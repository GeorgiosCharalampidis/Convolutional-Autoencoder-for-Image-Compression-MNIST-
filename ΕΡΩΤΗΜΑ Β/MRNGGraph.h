#ifndef PROJECT_K23_SEC_MRNGGRAPH_H
#define PROJECT_K23_SEC_MRNGGRAPH_H

#include <vector>
#include "global_functions.h" // Include the header for euclideanDistance

class MRNGNode {
public:
    std::vector<unsigned char> data;
    std::vector<MRNGNode*> neighbors;
    explicit MRNGNode(const std::vector<unsigned char>& data);
};

class MRNGGraph {
private:
    std::vector<MRNGNode> nodes;

public:
    explicit MRNGGraph(const std::vector<std::vector<unsigned char>>& dataset, int l = 20, int N = 1);
    std::vector<std::pair<int, double>> searchOnGraph(const std::vector<unsigned char>& query, int startNodeIndex, int k, int l);
    [[nodiscard]] const std::vector<MRNGNode>& getNodes() const { return nodes; }

};

#endif //PROJECT_K23_SEC_MRNGGRAPH_H

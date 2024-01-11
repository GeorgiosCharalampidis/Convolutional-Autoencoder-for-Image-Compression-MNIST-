#include "MRNGGraph.h"
#include <algorithm>

// Node constructor
MRNGNode::MRNGNode(const std::vector<unsigned char>& data) : data(data) {}

// MRNG Graph constructor
MRNGGraph::MRNGGraph(const std::vector<std::vector<unsigned char>>& dataset, int l, int N) {
    // Reserve memory for nodes to improve efficiency
    nodes.reserve(dataset.size());

    // Construct the graph from the dataset
    for (const auto& dataPoint : dataset) {
        nodes.emplace_back(dataPoint); // Adding each data point as a node
    }

    // MRNG construction
    for (auto& p : nodes) {
        std::vector<MRNGNode*> Rp; // Candidate set for potential neighbors
        Rp.reserve(nodes.size() - 1);

        // Exclude the current node when considering potential neighbors
        for (auto& r : nodes) {
            if (&p != &r) {
                Rp.push_back(&r);
            }
        }

        // Calculate distances and sort Rp according to distance to p
        std::vector<std::pair<MRNGNode*, double>> distRp; // Pair of node and its distance to p
        for (auto* node : Rp) {
            double dist = euclideanDistance(p.data, node->data); // Euclidean distance calculation
            distRp.emplace_back(node, dist);
        }

        // Sort based on distance - closer nodes first
        std::sort(distRp.begin(), distRp.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

        std::vector<MRNGNode*> Lp; // Selected set of neighbors
        Lp.reserve(l); // Reserve memory for efficiency

        // Initialize Lp with the closest N nodes to p in Rp
        for (int i = 0; i < std::min(N, static_cast<int>(distRp.size())); ++i) {
            Lp.push_back(distRp[i].first);
        }

        // Evaluate remaining nodes in Rp
        for (const auto& [r, dist_pr] : distRp) {
            // Skip if already in Lp or if Lp has enough neighbors
            if (std::find(Lp.begin(), Lp.end(), r) != Lp.end() || Lp.size() >= l) {
                continue;
            }

            // Ensuring Monotonic Path Validation
            bool isLongestEdge = true;

            for (auto& t : Lp) {
                double dist_pt = euclideanDistance(p.data, t->data);
                double dist_rt = euclideanDistance(r->data, t->data);

                // Check if the potential edge is the longest in the triangle p-r-t
                if (dist_pr <= dist_pt || dist_pr <= dist_rt) {
                    isLongestEdge = false;
                    break;
                }
            }

            // Add the edge if it satisfies the monotonic path condition
            if (isLongestEdge) {
                Lp.push_back(r);
            }
        }

        p.neighbors = Lp; // Set the neighbors for node p
    }
}

// Search function on the MRNG
std::vector<std::pair<int, double>> MRNGGraph::searchOnGraph(const std::vector<unsigned char>& query, int startNodeIndex, int k, int l) {
    std::vector<std::pair<int, double>> potentialNeighbors; // To store potential neighbors
    std::vector<MRNGNode*> candidatePool; // Pool of candidate nodes for the search
    const auto& nodes_ = this->getNodes(); // Get all nodes in the graph

    // Start from the specified node
    candidatePool.push_back(const_cast<MRNGNode*>(&nodes_[startNodeIndex]));

    // Search loop
    while (!candidatePool.empty() && candidatePool.size() < l) {
        auto currentNode = candidatePool.front();
        candidatePool.erase(candidatePool.begin());

        // Explore neighbors of the current node
        for (auto& neighbor : currentNode->neighbors) {
            double distance = euclideanDistance(query, neighbor->data); // Distance to query
            int nodeIndex = std::distance(nodes_.begin(), std::find_if(nodes_.begin(), nodes_.end(),
                                                                       [neighbor](const MRNGNode& node) {
                                                                           return &node == neighbor;
                                                                       }));
            potentialNeighbors.emplace_back(nodeIndex, distance);

            // Add neighbor to candidate pool if not already present
            if (std::find(candidatePool.begin(), candidatePool.end(), neighbor) == candidatePool.end()) {
                candidatePool.push_back(neighbor);
            }
        }

        // Sort and limit candidate pool size to l
        std::sort(candidatePool.begin(), candidatePool.end(), [&](MRNGNode* a, MRNGNode* b) {
            return euclideanDistance(query, a->data) < euclideanDistance(query, b->data);
        });
        if (candidatePool.size() > l) {
            candidatePool.resize(l);
        }
    }

    // Sort potential neighbors based on their distance to the query point
    std::sort(potentialNeighbors.begin(), potentialNeighbors.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second < b.second;
              });

    // Remove duplicates from the potential neighbors
    auto last = std::unique(potentialNeighbors.begin(), potentialNeighbors.end(),
                            [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                                return a.first == b.first;
                            });
    potentialNeighbors.erase(last, potentialNeighbors.end());

    // Keep only the top k elements as the result
    if (potentialNeighbors.size() > k) {
        potentialNeighbors.resize(k);
    }

    return potentialNeighbors;
}

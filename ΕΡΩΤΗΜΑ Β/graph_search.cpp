#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include "mnist.h"
#include "lsh_class.h"
#include "global_functions.h"
#include "graph.h"
#include "MRNGGraph.h"

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);

    // Δηλώσεις μεταβλητών για τα αρχεία εισόδου
    std::string inputFile784, queryFile784, inputFile20, queryFile20, outputFile;

    int number_of_images, image_size;
    int k = 50; // Number of nearest neighbors in graph
    int E = 30; // Number of expansions
    int R = 12; // Number of random restarts
    int N = 320;  // Number of nearest neighbors to search for
    int l = 320;  // Only for Search-on-Graph
    int mode = 0; // 1 for true, 2 for GNNS, 3 for MRNG


    // Παρσάρισμα Παραμέτρων
    for (size_t i = 1; i < args.size(); i++) {
        if (args[i] == "-d784") {
            inputFile784 = args[++i];
        } else if (args[i] == "-q784") {
            queryFile784 = args[++i];
        } else if (args[i] == "-d20") {
            inputFile20 = args[++i];
        } else if (args[i] == "-q20") {
            queryFile20 = args[++i];
        } else if (args[i] == "-o") {
            outputFile = args[++i];
        } else if (args[i] == "-m") {
            mode = std::stoi(args[++i]);
        }
    }
    // Read the dataset and query set
    std::vector<std::vector<unsigned char>> dataset_784 = read_mnist_images(inputFile784, number_of_images, image_size);
    std::vector<std::vector<unsigned char>> dataset_20 = read_mnist_images(inputFile20, number_of_images, image_size);

    std::vector<std::vector<unsigned char>> query_set_784 = read_mnist_images(queryFile784, number_of_images, image_size);
    std::vector<std::vector<unsigned char>> query_set_20 = read_mnist_images(queryFile20, number_of_images, image_size);

    std::ofstream outputFileStream(outputFile);

    double totalTAlgorithm = 0.0;
    double totalTTrue = 0.0;
    double maxApproximationFactor = 0.0;
    double distance_Up_784;
    double trueResults_Up_784 = 0.0;
    double AF = 0.0;
    double MAF = 0.0;

    std::vector<std::vector<unsigned char>> testset_784,testset_20;
    // testset with 3000 images of dataset
    for (int i = 0; i < 3000; ++i) {
        testset_784.push_back(dataset_784[i]);
        testset_20.push_back(dataset_20[i]);
    }

    if (mode == 1) {

        outputFileStream << "Exhaustive Search Results" << std::endl;

        for (int i = 0; i < 10; ++i) {
            outputFileStream << "\nQuery: " << i << std::endl;

            auto startTimeAlgorithm = std::chrono::high_resolution_clock::now();
            auto results = trueNNearestNeighbors(testset_20, query_set_20[i], N);
            auto endTimeAlgorithm = std::chrono::high_resolution_clock::now();

            auto startTimeTrue = std::chrono::high_resolution_clock::now();
            auto trueResults_784 = trueNNearestNeighbors(testset_784, query_set_784[i], N);
            auto endTimeTrue = std::chrono::high_resolution_clock::now();

            double tAlgorithm = std::chrono::duration<double, std::milli>(endTimeAlgorithm - startTimeAlgorithm).count() / 1000.0;
            double tTrue = std::chrono::duration<double, std::milli>(endTimeTrue - startTimeTrue).count() / 1000.0;
            for (int j = 0; j < N; ++j) {
                // count the distance
                trueResults_Up_784 = euclideanDistance(testset_784[results[j].first], query_set_784[i]);
                AF = trueResults_Up_784 / trueResults_784[j].second;
                outputFileStream << "Nearest neighbor-" << j + 1 << ": " << results[j].first << std::endl;
                outputFileStream << "distanceUp_784: " << trueResults_Up_784 << std::endl;
                outputFileStream << "distanceTrue_784: " << trueResults_784[j].second << std::endl;
            }

            totalTAlgorithm += tAlgorithm;
            totalTTrue += tTrue;
            MAF += AF;

        }
        // Calculate the average values for the 10 queries
        totalTAlgorithm /= 10;
        totalTTrue /= 10;
        MAF /= 10;

        outputFileStream << std::endl;
        outputFileStream << "tTrue_Up_784: " << totalTAlgorithm << std::endl;
        outputFileStream << "MAF: " << MAF << std::endl;
        std::cout << "tTrue_Up_784: " << totalTAlgorithm << std::endl;
        //std::cout << "totalTTrue: " << totalTTrue << std::endl;
        std::cout << "MAF: " << MAF << std::endl;


    }   else if (mode == 2) {

        int T = 10; // Number of greedy steps

        LSH lsh(testset_20, image_size, 4, 5, 1, 10000);

        std::cout << "Started building the k-NNG..." << std::endl;
        Graph kNNG_L_20 = buildKNNG(lsh, k, testset_20.size());
        std::cout << "Finished building the k-NNG." << std::endl;

        outputFileStream << "GNNS Results" << std::endl;

        // 20 dataset
        for (int i = 0; i < 10; ++i) {
            outputFileStream << "\nQuery: " << i << std::endl;

            auto startTimeAlgorithm = std::chrono::high_resolution_clock::now();
            auto results = kNNG_L_20.GNNS(query_set_20[i], N, R, T, E);
            auto endTimeAlgorithm = std::chrono::high_resolution_clock::now();

            auto trueResults_784 = trueNNearestNeighbors(testset_784, query_set_784[i], N);



            double tAlgorithm = std::chrono::duration<double, std::milli>(endTimeAlgorithm - startTimeAlgorithm).count() / 1000.0;


            for (int j = 0; j < N; ++j) {
                // count the distance
                distance_Up_784 = euclideanDistance(testset_784[results[j].first], query_set_784[i]);
                AF = distance_Up_784 / trueResults_784[j].second;

                outputFileStream << "Nearest neighbor-" << j + 1 << ": " << results[j].first << std::endl;
                outputFileStream << "distanceUp_784: " << distance_Up_784 << std::endl;
                outputFileStream << "distanceTrue_784: " << trueResults_784[j].second << std::endl;

            }

            totalTAlgorithm += tAlgorithm;
            MAF += AF;

        }
        // Calculate the average values for the 10 queries

        totalTAlgorithm /= 10;
        MAF /= 10;

        outputFileStream << std::endl;
        outputFileStream << "tAverageApproximate: " << totalTAlgorithm << std::endl;
        outputFileStream << "MAF: " << MAF << std::endl;
        std::cout << "tAverageApproximate: " << totalTAlgorithm << std::endl;
        std::cout << "MAF: " << MAF << std::endl;

    }   else if (mode==3) {


        std::cout << "Started building the MRNG..." << std::endl;
        MRNGGraph mrngGraph(testset_20, l, N);
        std::cout << "Finished building the MRNG." << std::endl;

        outputFileStream << "MRNG Results" << std::endl;

        for (int i = 0; i < 10; ++i) {
            outputFileStream << "\nQuery: " << i << std::endl;

            // Start time for MRNG algorithm
            auto startTimeAlgorithm = std::chrono::high_resolution_clock::now();
            auto results = mrngGraph.searchOnGraph(query_set_20[i], 0, N, l);
            auto endTimeAlgorithm = std::chrono::high_resolution_clock::now();

            auto trueResults_784 = trueNNearestNeighbors(testset_784, query_set_784[i], N);

            // Calculate the time taken by the MRNG algorithm
            double tAlgorithm = std::chrono::duration<double, std::milli>(endTimeAlgorithm - startTimeAlgorithm).count() / 1000.0;

            for (int j = 0; j < N; ++j) {
                // count the distance

                distance_Up_784 = euclideanDistance(testset_784[results[j].first], query_set_784[i]);
                AF = distance_Up_784 / trueResults_784[j].second;
                outputFileStream << "Nearest neighbor-" << j + 1 << ": " << results[j].first << std::endl;
                outputFileStream << "distanceUp_784: " << distance_Up_784 << std::endl;
                outputFileStream << "distanceTrue_784: " << trueResults_784[j].second << std::endl;
            }
            totalTAlgorithm += tAlgorithm;
            MAF += AF;
        }
        // Calculate the average values for the 10 queries
        totalTAlgorithm /= 10;
        MAF /= 10;

        outputFileStream << std::endl;
        outputFileStream << "tAverageApproximate: " << totalTAlgorithm << std::endl;
        outputFileStream << "MAF: " << MAF << std::endl;
        std::cout << "tAverageApproximate: " << totalTAlgorithm << std::endl;
        std::cout << "MAF: " << MAF << std::endl;


    }

    return 0;
}



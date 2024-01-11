#ifndef MNIST_H
#define MNIST_H

#include <vector>
#include <string>

// Function declarations (signatures)
std::vector<std::vector<unsigned char>> read_mnist_images(const std::string& full_path, int& number_of_images, int& image_size);
void print_image(const std::vector<unsigned char>& image, int width, int height);

#endif

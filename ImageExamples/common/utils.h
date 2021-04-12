#pragma once

#include <string>

struct PPMImage {
    std::string header;
    int width, height;
    int maxColor;
    unsigned char* data;
};

std::istream& operator >>(std::istream& inputStream, PPMImage& other);
std::ostream& operator <<(std::ostream& outputStream, const PPMImage& other);
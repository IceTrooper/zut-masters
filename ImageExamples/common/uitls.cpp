#include "utils.h"
#include <fstream>

std::istream& operator >>(std::istream& inputStream, PPMImage& other)
{
    inputStream >> other.header;
    inputStream >> other.width >> other.height >> other.maxColor;
    //inputStream.get(); // skip the trailing white space
    //size_t size = other.width * other.height * 3;
    size_t size = other.width * other.height * 3;
    //other.ptr = new unsigned char[size];
    //other.data = new unsigned char[other.width * other.height * 3];
    other.data = (unsigned char*)malloc(size);
    inputStream.read((char*)other.data, size);
    return inputStream;
}

std::ostream& operator <<(std::ostream& outputStream, const PPMImage& other)
{
    outputStream << "P6" << "\n"
        << other.width << " "
        << other.height << "\n"
        << other.maxColor << "\n"
        ;
    size_t size = other.width * other.height * 3;
    //outputStream.write(other.data, size);
    return outputStream;
}
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

// Use opencl.hpp instead of cl2.hpp to make it clear that it supports all versions of OpenCL
// #include <CL/cl2.hpp>
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "../common/utils.h"
#include "../common/CImg.h"

#define VERBOSE true

using namespace std;
using namespace cimg_library;

bool CheckPreferredPlatformMatch(cl::Platform platform, const string preferredPlatform)
{
	string platformName = platform.getInfo<CL_PLATFORM_NAME>();
	if (platformName.find(preferredPlatform) == string::npos)
	{
		return false;
	}

	cout << "Platform: " << platformName << "\n";
	return true;
}

cl::Platform FindOpenCLPlatform()
{
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	cout << "Number of available platforms: " << platforms.size() << "\n";

	for (auto& platform : platforms)
	{
		if (!CheckPreferredPlatformMatch(platform, "Intel"))
		{
			continue;
		}

		vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

		if (devices.size() != 0)
		{
			cout << "Required device was found.\n";
			cout << "Number of available devices: " << devices.size() << "\n";

			return platform;
		}
	}
	throw runtime_error("Required device was not found on any platform!");
}

void Profile(cl::Event& clEvent)
{
	cl_ulong startTime = clEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong endTime = clEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	cl_ulong elapsed = endTime - startTime;
	cout << "Time elapsed: " << elapsed << " ns\n";
}

int Program(int argc, char* argv[])
{
	const string filename = "imageScaling";
	const string extension = ".ppm";

	const int destinationWidth = 800;
	const int destinationHeight = 800;
	const float widthNormFactor = 1.0f / destinationWidth;
	const float heightNormFactor = 1.0f / destinationHeight;

	CImg<unsigned char> inputImage((filename + extension).c_str());
	CImgDisplay inputDisplay(inputImage, "Input image");
	const int imageWidth = inputImage.width();
	const int imageHeight = inputImage.height();
	// Add alpha channel to PPM image, because OpenCL can't use CL_RGB with CL_UNSIGNED_INT8
	// https://stackoverflow.com/questions/32238522/set-default-value-for-alpha-in-cimg
	inputImage.channels(0, 3);
	inputImage.get_shared_channel(3).fill(255);
	inputImage.permute_axes("cxyz");

	cl::Platform platform = FindOpenCLPlatform();

	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, contextProperties);
	vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	cl::Device device = devices[0];
	cl::CommandQueue commandQueue(context, device, cl::QueueProperties::Profiling);

	// Read source file
	ifstream sourceFile("ImageScaling.cl");
	string kernelSource(
		istreambuf_iterator<char>(sourceFile),
		(istreambuf_iterator<char>()));
	cl::Program::Sources source{ kernelSource };
	cl::Program program = cl::Program(context, source);
	// Build binary version of program.
	program.build(device);

	cout << "\n\nImage scaling\n";

	cl::ImageFormat imageFormat(CL_RGBA, CL_UNSIGNED_INT8);
	cl::Image2D clImageIn(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, (void*)inputImage.data(), NULL);
	cl::Image2D clImageOut(context, CL_MEM_WRITE_ONLY, imageFormat, destinationWidth, destinationHeight, 0, NULL, NULL);

	cl::Kernel kernel(program, "ImageScaling");

	kernel.setArg(0, clImageIn);
	kernel.setArg(1, clImageOut);
	kernel.setArg(2, widthNormFactor);
	kernel.setArg(3, heightNormFactor);

	cl::NDRange global(destinationWidth, destinationHeight);

	cl::Event clEvent;
	commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, NULL, &clEvent);
	commandQueue.finish();

	Profile(clEvent);

	array<size_t, 3> origin = { 0, 0, 0 };
	array<size_t, 3> region = { destinationWidth, destinationHeight, 1 };

	// Reading image
	CImg<unsigned char> outputImage(destinationWidth, destinationHeight, 1, 4);
	outputImage.permute_axes("cxyz");

	commandQueue.enqueueReadImage(clImageOut, CL_TRUE, origin, region, 0, 0, (void*)outputImage.data());

	outputImage.permute_axes("yzcx");
	outputImage.channels(0, 2);
	outputImage.save((filename + "Out" + extension).c_str());

	// Show images
	if (VERBOSE)
	{
		CImgDisplay outputDisplay(outputImage, "Output image");
		while (!inputDisplay.is_closed() && !outputDisplay.is_closed())
		{
			inputDisplay.wait();
		}
	}

	return 0;
}

int main(int argc, char* argv[])
{
	try
	{
		Program(argc, argv);
	}
	catch (cl::Error e)
	{
		cout << "Returned code (" << e.err() << ": " << OCL_GetErrorString(e.err()) << "): " << e.what() << "\n";
		return e.err();
	}
	catch (const exception& e)
	{
		cout << "Error: " << e.what() << "\n";
		return -1;
	}
	catch (...)
	{
		cout << "Unknown exception\n";
		return -128;
	}

	system("pause");
	return 0;
}
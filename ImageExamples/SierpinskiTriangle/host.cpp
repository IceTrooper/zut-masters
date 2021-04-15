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
	cl::Device device;
	cl::Program program;

	try
	{
		const cl_uint maxLevel = 10;
		const cl_uint imageSize = 1024;

		CImg<unsigned char> outputImage(imageSize, imageSize, 1, 4);
		outputImage.permute_axes("cxyz");

		cl::Platform platform = FindOpenCLPlatform();

		cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
		cl::Context context(CL_DEVICE_TYPE_GPU, contextProperties);
		vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		device = devices[0];
		cl::CommandQueue commandQueue(context, device, cl::QueueProperties::Profiling);

		cl_int err = CL_SUCCESS;
		// Use this:
		//cl_command_queue_properties props = CL_QUEUE_ON_DEVICE_DEFAULT;
		//cl::DeviceCommandQueue deviceCommandQueue(context, device, (cl::DeviceQueueProperties)props, &err);
		// or this:
		// Device side queue - 16MB
		cl::DeviceCommandQueue deviceCommandQueue(context, device, (cl_uint)(16 * 1024 * 1024), (cl::DeviceQueueProperties)CL_QUEUE_ON_DEVICE_DEFAULT, &err);
		cout << "DeviceCommandQueue return status: " << err << "\n";

		// Read source file
		ifstream sourceFile("SierpinskiTriangle.cl");
		string kernelSource(
			istreambuf_iterator<char>(sourceFile),
			(istreambuf_iterator<char>()));
		cl::Program::Sources source{ kernelSource };
		program = cl::Program(context, source);
		// Build binary version of program.
		program.build(device, "-cl-std=CL2.0");

		cout << "\n\nSierpinski Triangle:\n";

		cl::ImageFormat imageFormat(CL_RGBA, CL_UNSIGNED_INT8);
		cl::Image2D clImageOut(context, CL_MEM_WRITE_ONLY, imageFormat, imageSize, imageSize, 0, NULL, NULL);

		cl::Kernel kernel(program, "Triangle");

		cl_uint level = 0;
		cl_uint xOffset = 0;
		cl_uint yOffset = 0;
		kernel.setArg(0, clImageOut);
		kernel.setArg(1, sizeof(cl_uint), &imageSize);
		kernel.setArg(2, sizeof(cl_uint), &xOffset);
		kernel.setArg(3, sizeof(cl_uint), &yOffset);
		kernel.setArg(4, sizeof(cl_uint), &level);
		kernel.setArg(5, sizeof(cl_uint), &maxLevel);

		cl::NDRange global(1);

		cl::Event clEvent;
		commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, NULL, &clEvent);
		commandQueue.finish();

		Profile(clEvent);

		array<size_t, 3> origin = { 0, 0, 0 };
		array<size_t, 3> region = { imageSize, imageSize, 1 };

		commandQueue.enqueueReadImage(clImageOut, CL_TRUE, origin, region, 0, 0, (void*)outputImage.data());

		outputImage.permute_axes("yzcx");
		outputImage.channels(0, 2);
		outputImage.save(("sierpinskiTriangle-" + to_string(imageSize) + ".ppm").c_str());

		// Show images
		if (VERBOSE)
		{
			CImgDisplay outputDisplay(outputImage, "Output image");
			while (!outputDisplay.is_closed())
			{
				outputDisplay.wait();
			}
		}
	}
	catch (cl::Error e)
	{
		cout << "Returned code (" << e.err() << ": " << OCL_GetErrorString(e.err()) << "): " << e.what() << "\n";

		if (e.err() == CL_BUILD_PROGRAM_FAILURE)
		{
			cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
			if (status == CL_BUILD_ERROR)
			{
				string name = device.getInfo<CL_DEVICE_NAME>();
				string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
				cout << "Build log for " << name << ":" << "\n" << buildlog << "\n";
			}
		}

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

	return 0;
}

int main(int argc, char* argv[])
{

	int status = Program(argc, argv);

	system("pause");
	return status;
}
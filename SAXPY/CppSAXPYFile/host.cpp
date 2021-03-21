#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

// Use opencl.hpp instead of cl2.hpp to make it clear that it supports all versions of OpenCL
// #include <CL/cl2.hpp>
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;

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
			cout << "Number of available devices: " << devices.size();

			return platform;
		}
	}
	throw runtime_error("Required device was not found on any platform!");
}

void FillOrdered(cl_float* floatArray, cl_uint n, float start, float step)
{
	for (int i = 0; i < n; i++)
	{
		floatArray[i] = start + i * step;
	}
}

void FillRandom(cl_float* floatArray, cl_uint n)
{
	srand(12345);
	for (int i = 0; i < n; i++)
	{
		floatArray[i] = rand();
	}
}

void FillEmpty(cl_float* floatArray, cl_uint n)
{
	for (int i = 0; i < n; i++)
	{
		floatArray[i] = 0.0f;
	}
}

int Program(int argc, char* argv[])
{
	cl::Platform platform = FindOpenCLPlatform();

	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, contextProperties);
	vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	cl::Device device = devices[0];

	const int N = 32;
	size_t nBytes = N * sizeof(float);
	const float inputA = 2.5f;
	float* hostInputX = (float*)malloc(nBytes);
	float* hostInputY = (float*)malloc(nBytes);
	float* hostOutZ = (float*)malloc(nBytes);
	FillOrdered(hostInputX, N, 1.0f, 1.0f);
	FillOrdered(hostInputY, N, 1.0f, 2.0f);
	FillEmpty(hostOutZ, N);

	cl::Buffer deviceInX(context, CL_MEM_READ_ONLY, nBytes);
	cl::Buffer deviceInY(context, CL_MEM_READ_ONLY, nBytes);
	cl::Buffer deviceOutZ(context, CL_MEM_WRITE_ONLY, nBytes);

	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	cl::CommandQueue commandQueue(context, device, properties);

	commandQueue.enqueueWriteBuffer(deviceInX, true, 0, nBytes, (void*)hostInputX);
	commandQueue.enqueueWriteBuffer(deviceInY, true, 0, nBytes, (void*)hostInputY);

	// Read source file
	ifstream sourceFile("SAXPY.cl");
	string kernelSource(
		istreambuf_iterator<char>(sourceFile),
		(istreambuf_iterator<char>()));
	cl::Program::Sources source{ kernelSource };
	cl::Program program = cl::Program(context, source);
	// Build binary version of program.
	program.build(device);

	cl::Kernel kernel(program, "Saxpy");

	kernel.setArg(0, inputA);
	kernel.setArg(1, deviceInX);
	kernel.setArg(2, deviceInY);
	kernel.setArg(3, deviceOutZ);
	kernel.setArg(4, sizeof(int), &N);

	cl::Event event;
	commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N, 1), cl::NullRange, NULL, &event);
	event.wait();

	commandQueue.enqueueReadBuffer(deviceOutZ, true, 0, nBytes, (void*)hostOutZ);
	for (int i = 0; i < N; i++)
	{
		cout << hostOutZ[i] << " ";
	}
	cout << endl;

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
		cout << "Returned code (" << e.err() << "): " << e.what() << "\n";
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
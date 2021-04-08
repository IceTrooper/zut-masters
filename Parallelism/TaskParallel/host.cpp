/*
* Task paralell example.
*/

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

// Use opencl.hpp instead of cl2.hpp to make it clear that it supports all versions of OpenCL
// #include <CL/cl2.hpp>
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>

#define RAND_BASE 10
#define ROW_COUNT 1024
#define VERBOSE false

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
			cout << "Number of available devices: " << devices.size() << "\n";

			return platform;
		}
	}
	throw runtime_error("Required device was not found on any platform!");
}

void FillOrdered(cl_float* matrix, cl_uint n, cl_uint m, float start, float step)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			matrix[i * m + j] = start + (i * m + j) * step;
			//matrix[i, j] = start + (i * m + j) * step;
		}
	}
}

void FillRandom(cl_float* matrix, cl_uint n, cl_uint m, bool normalized = false)
{
	srand(2021);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			matrix[i * m + j] = rand() / (float)RAND_MAX;
			if (!normalized)
			{
				matrix[i] *= RAND_BASE;
			}
		}
	}
}

void FillEmpty(cl_float* matrix, cl_uint n, cl_uint m)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			matrix[i * m + j] = 0.0f;
		}
	}
}

void PrintMatrix(const float* matrix, const int nDim, const int mDim)
{
	for (int i = 0; i < nDim; i++)
	{
		for (int j = 0; j < mDim; j++)
		{
			cout << matrix[i * mDim + j] << " ";
		}
		cout << "\n";
	}
	cout << endl;
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
	const cl_uint row_count = ROW_COUNT;
	const cl_uint col_count = 3;
	const cl_uint count = row_count * col_count;

	cl_float* A = new cl_float[count];
	cl_float* B = new cl_float[count];
	cl_float* C = new cl_float[count];
	size_t sizeMat = count * sizeof(cl_float);

	cl::Platform platform = FindOpenCLPlatform();

	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, contextProperties);
	vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	cl::Device device = devices[0];
	cl::CommandQueue commandQueue(context, device, cl::QueueProperties::Profiling);

	// Read source file
	ifstream sourceFile("TaskParallel.cl");
	string kernelSource(
		istreambuf_iterator<char>(sourceFile),
		(istreambuf_iterator<char>()));
	cl::Program::Sources source{ kernelSource };
	cl::Program program = cl::Program(context, source);
	// Build binary version of program.
	program.build(device);

	cout << "\n\nParallelism - Task parallel example\n";

	FillOrdered(A, row_count, col_count, 1.0f, 1.0f);
	FillRandom(B, row_count, col_count, true);
	FillEmpty(C, row_count, col_count);
	if (VERBOSE)
	{
		PrintMatrix(A, row_count, col_count);
		PrintMatrix(B, row_count, col_count);
	}

	cout << "\n";

	cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeMat);
	cl::Buffer bufferB(context, CL_MEM_READ_ONLY, sizeMat);
	cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeMat);

	commandQueue.enqueueWriteBuffer(bufferA, true, 0, sizeMat, (void*)A);
	commandQueue.enqueueWriteBuffer(bufferB, true, 0, sizeMat, (void*)B);

	cl::Kernel kernels[3]{
		cl::Kernel(program, "TaskParallelAdd"),
		cl::Kernel(program, "TaskParallelSub"),
		cl::Kernel(program, "TaskParallelMul")
	};

	for (auto k : kernels)
	{
		k.setArg(0, bufferA);
		k.setArg(1, bufferB);
		k.setArg(2, bufferC);
		k.setArg(3, sizeof(cl_uint), &row_count);
	}

	for (auto k : kernels)
	{
		// commandQueue.enqueueTask is deprecated, use this instead:
		commandQueue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1), cl::NDRange(1), NULL, NULL);
	}

	commandQueue.finish();

	commandQueue.enqueueReadBuffer(bufferC, true, 0, sizeMat, (void*)C);

	if (VERBOSE)
	{
		PrintMatrix(C, row_count, col_count);
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

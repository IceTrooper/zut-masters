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

void FillRandom(cl_float* matrix, cl_uint n, cl_uint m)
{
	srand(12345);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			matrix[i * m + j] = rand();
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

void SgemmNaive(const int nDim, const int mDim, const int kDim, const float* A, const float* B, float* C)
{
	int i, j, k;
	float acc;

	for (i = 0; i < nDim; i++)
	{
		for (j = 0; j < mDim; j++)
		{
			acc = 0.0f;
			for (k = 0; k < kDim; k++)
			{
				acc += *(A + (i * nDim + k)) + *(B + (k * kDim + j));
			}
			*(C + (i * nDim + j)) = acc;
		}
	}
}

int Program(int argc, char* argv[])
{
	cl::Platform platform = FindOpenCLPlatform();

	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, contextProperties);
	vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	cl::Device device = devices[0];

	//const cl_uint threadBlockSize = 32;
	const cl_uint nDim = 16;//1024;
	const cl_uint kDim = 32;//2048;
	const cl_uint mDim = 64;//4096;

	cl_float* A = new cl_float[nDim * kDim];
	cl_float* B = new cl_float[kDim * mDim];
	cl_float* C = new cl_float[nDim * mDim];
	FillOrdered(A, nDim, kDim, 1.0f, 1.0f);
	FillOrdered(B, kDim, mDim, 1.0f, 2.0f);
	FillEmpty(C, nDim, mDim);

	//const int N = 32;
	//size_t nBytes = N * sizeof(float);
	//const float inputA = 2.5f;
	//float* hostInputX = (float*)malloc(nBytes);
	//float* hostInputY = (float*)malloc(nBytes);
	//float* hostOutZ = (float*)malloc(nBytes);
	//FillOrdered(hostInputX, N, 1.0f, 1.0f);
	//FillOrdered(hostInputY, N, 1.0f, 2.0f);
	//FillEmpty(hostOutZ, N);

	cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeof(A));
	cl::Buffer bufferB(context, CL_MEM_READ_ONLY, sizeof(B));
	cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(C));

	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	cl::CommandQueue commandQueue(context, device, properties);

	commandQueue.enqueueWriteBuffer(bufferA, true, 0, sizeof(A), (void*)A);
	commandQueue.enqueueWriteBuffer(bufferB, true, 0, sizeof(B), (void*)B);

	// Read source file
	ifstream sourceFile("SGEMM_naive.cl");
	string kernelSource(
		istreambuf_iterator<char>(sourceFile),
		(istreambuf_iterator<char>()));
	cl::Program::Sources source{ kernelSource };
	cl::Program program = cl::Program(context, source);
	// Build binary version of program.
	program.build(device);

	cl::Kernel kernel(program, "Sgemm");

	kernel.setArg(0, nDim);
	kernel.setArg(1, kDim);
	kernel.setArg(2, nDim);
	kernel.setArg(3, bufferA);
	kernel.setArg(4, bufferB);
	kernel.setArg(5, bufferC);
	//kernel.setArg(6, threadBlockSize);

	//cl::NDRange global = cl::NDRange(nDim, mDim);
	//cl::NDRange local = cl::NDRange(threadBlockSize, threadBlockSize);
	cl::NDRange global = cl::NDRange(nDim, mDim);
	cl::Event clEvent;
	//commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL, &event);
	commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, NULL, &clEvent);
	clEvent.wait();

	cl_ulong startTime = clEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong endTime = clEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	cl_ulong elapsed = endTime - startTime;

	cout << "\n\nTime elapsed: " << elapsed << "\n";

	commandQueue.enqueueReadBuffer(bufferC, true, 0, sizeof(C), (void*)C);
	for (int i = 0; i < nDim; i++)
	{
		for (int j = 0; j < mDim; j++)
		{
			cout << C[i * mDim + j] << " ";
		}
		cout << "\n";
	}
	cout << endl;

	cout << "\nCheck:\n";
	// write more kernels in one file

	delete[] A;
	delete[] B;
	delete[] C;

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
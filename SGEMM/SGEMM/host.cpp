#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

// Use opencl.hpp instead of cl2.hpp to make it clear that it supports all versions of OpenCL
// #include <CL/cl2.hpp>
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "host.h"

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
				acc += *(A + (i * kDim + k)) * *(B + (k * mDim + j));
			}
			*(C + (i * mDim + j)) = acc;
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

void KernelSgemmNaive(cl::Program& program, cl::CommandQueue& commandQueue,
	const cl_uint nDim, const cl_uint kDim, const cl_uint mDim,
	cl::Buffer& bufferA, cl::Buffer& bufferB, cl::Buffer& bufferC)
{
	cl::Kernel kernel(program, "Sgemm_simple");

	kernel.setArg(0, sizeof(cl_uint), &nDim);
	kernel.setArg(1, sizeof(cl_uint), &kDim);
	kernel.setArg(2, sizeof(cl_uint), &mDim);
	kernel.setArg(3, bufferA);
	kernel.setArg(4, bufferB);
	kernel.setArg(5, bufferC);

	cl::NDRange global = cl::NDRange(nDim, mDim);
	cl::Event clEvent;

	commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, NULL, &clEvent);
	clEvent.wait();

	Profile(clEvent);
}

void KernelSgemmComputeUnits(cl::Device& device, cl::Program& program, cl::CommandQueue& commandQueue,
	const cl_uint nDim, const cl_uint kDim, const cl_uint mDim,
	cl::Buffer& bufferA, cl::Buffer& bufferB, cl::Buffer& bufferC)
{
	cl_uint maxComputeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << maxComputeUnits << "\n";
	if (nDim % maxComputeUnits != 0)
	{
		throw runtime_error("nDim must be divisible by the CL_DEVICE_MAX_COMPUTE_UNITS without reminder!");
	}

	cl::Kernel kernel(program, "Sgemm_compute_units");

	kernel.setArg(0, sizeof(cl_uint), &nDim);
	kernel.setArg(1, sizeof(cl_uint), &kDim);
	kernel.setArg(2, sizeof(cl_uint), &mDim);
	kernel.setArg(3, bufferA);
	kernel.setArg(4, bufferB);
	kernel.setArg(5, bufferC);
	cout << kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << "\n";

	cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
	cout << "CL_KERNEL_WORK_GROUP_SIZE: " << kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << "\n";

	cl::NDRange global = cl::NDRange(nDim);
	cl::NDRange local = cl::NDRange(nDim/maxComputeUnits);
	cl::Event clEvent;

	commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL, &clEvent);
	clEvent.wait();

	Profile(clEvent);
}

// This is the same code as KernelSgemmComputeUnits but with different kernel.
void KernelSgemmPrivate(cl::Device& device, cl::Program& program, cl::CommandQueue& commandQueue,
	const cl_uint nDim, const cl_uint kDim, const cl_uint mDim,
	cl::Buffer& bufferA, cl::Buffer& bufferB, cl::Buffer& bufferC)
{
	cl_uint maxComputeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << maxComputeUnits << "\n";
	if (nDim % maxComputeUnits != 0)
	{
		throw runtime_error("nDim must be divisible by the CL_DEVICE_MAX_COMPUTE_UNITS without reminder!");
	}

	cl::Kernel kernel(program, "Sgemm_private");

	kernel.setArg(0, sizeof(cl_uint), &nDim);
	kernel.setArg(1, sizeof(cl_uint), &kDim);
	kernel.setArg(2, sizeof(cl_uint), &mDim);
	kernel.setArg(3, bufferA);
	kernel.setArg(4, bufferB);
	kernel.setArg(5, bufferC);
	cout << kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << "\n";

	cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
	cout << "CL_KERNEL_WORK_GROUP_SIZE: " << kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << "\n";

	cl::NDRange global = cl::NDRange(nDim);
	cl::NDRange local = cl::NDRange(nDim / maxComputeUnits);
	cl::Event clEvent;

	commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL, &clEvent);
	clEvent.wait();

	Profile(clEvent);
}

void KernelSgemmLocal(cl::Device& device, cl::Program& program, cl::CommandQueue& commandQueue,
	const cl_uint nDim, const cl_uint kDim, const cl_uint mDim,
	cl::Buffer& bufferA, cl::Buffer& bufferB, cl::Buffer& bufferC)
{
	cl_uint maxComputeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << maxComputeUnits << "\n";
	if (nDim % maxComputeUnits != 0)
	{
		throw runtime_error("nDim must be divisible by the CL_DEVICE_MAX_COMPUTE_UNITS without reminder!");
	}

	cl::Kernel kernel(program, "Sgemm_local");

	kernel.setArg(0, sizeof(cl_uint), &nDim);
	kernel.setArg(1, sizeof(cl_uint), &kDim);
	kernel.setArg(2, sizeof(cl_uint), &mDim);
	kernel.setArg(3, bufferA);
	kernel.setArg(4, bufferB);
	kernel.setArg(5, bufferC);
	kernel.setArg(6, kDim * sizeof(float), NULL);

	cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
	cout << "CL_KERNEL_WORK_GROUP_SIZE: " << kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << "\n";

	cl::NDRange global = cl::NDRange(nDim);
	cl::NDRange local = cl::NDRange(nDim / maxComputeUnits);
	cl::Event clEvent;

	commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL, &clEvent);
	clEvent.wait();

	Profile(clEvent);
}

int Program(int argc, char* argv[])
{
	cl::Platform platform = FindOpenCLPlatform();

	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, contextProperties);
	vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	cl::Device device = devices[0];

	cout << "\n";
	const cl_uint nDim = N_DIM;
	const cl_uint kDim = K_DIM;
	const cl_uint mDim = M_DIM;

	cout << "N: " << nDim << ", K: " << kDim << ", M: " << mDim << "\n";

	cl_float* A = new cl_float[nDim * kDim];
	cl_float* B = new cl_float[kDim * mDim];
	cl_float* C = new cl_float[nDim * mDim];
	size_t sizeA = nDim * kDim * sizeof(float);
	size_t sizeB = kDim * mDim * sizeof(float);
	size_t sizeC = nDim * mDim * sizeof(float);

	FillOrdered(A, nDim, kDim, 0.00001f, 0.00001f);
	FillOrdered(B, kDim, mDim, 0.00002f, 0.00002f);
	FillEmpty(C, nDim, mDim);

	// Printing matrices to test out.
	if (VERBOSE)
	{
		//PrintMatrix(A, nDim, kDim);
		//PrintMatrix(B, kDim, mDim);
		PrintMatrix(C, nDim, mDim);
	}

	cl_float* hostC;
	// Host multiplication
	if (COMPUTE_HOST)
	{
		hostC = new cl_float[nDim * mDim];
		//FillEmpty(hostC, nDim, mDim);
		std::memcpy(hostC, C, sizeC);
		cout << "Naive host matrix multiplication:\n";
		auto tStart = chrono::high_resolution_clock::now();
		SgemmNaive(nDim, mDim, kDim, A, B, hostC);
		auto tEnd = chrono::high_resolution_clock::now();

		auto ns_int = chrono::duration_cast<chrono::nanoseconds>(tEnd - tStart);
		cout << "Naive time elapsed: " << ns_int.count() << " ns\n";

		if (VERBOSE) PrintMatrix(hostC, nDim, mDim);
	}

	cout << "Kernel matrix multiplication:\n";

	cl::Buffer bufferA(context, CL_MEM_READ_ONLY, sizeA);
	cl::Buffer bufferB(context, CL_MEM_READ_ONLY, sizeB);
	cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeC);

	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	cl::CommandQueue commandQueue(context, device, properties);

	commandQueue.enqueueWriteBuffer(bufferA, true, 0, sizeA, (void*)A);
	commandQueue.enqueueWriteBuffer(bufferB, true, 0, sizeB, (void*)B);

	// Read source file
	ifstream sourceFile("SGEMM.cl");
	string kernelSource(
		istreambuf_iterator<char>(sourceFile),
		(istreambuf_iterator<char>()));
	cl::Program::Sources source{ kernelSource };
	cl::Program program = cl::Program(context, source);
	// Build binary version of program.
	program.build(device);

	// Main kernel program
	//KernelSgemmNaive(program, commandQueue, nDim, kDim, mDim, bufferA, bufferB, bufferC);
	//KernelSgemmComputeUnits(device, program, commandQueue, nDim, kDim, mDim, bufferA, bufferB, bufferC);
	//KernelSgemmPrivate(device, program, commandQueue, nDim, kDim, mDim, bufferA, bufferB, bufferC);
	KernelSgemmLocal(device, program, commandQueue, nDim, kDim, mDim, bufferA, bufferB, bufferC);

	// Read and check results
	commandQueue.enqueueReadBuffer(bufferC, true, 0, sizeC, (void*)C);
	if (VERBOSE)
	{
		PrintMatrix(C, nDim, mDim);
	}

	if (COMPUTE_HOST)
	{
		bool isEqual = true;
		for (size_t i = 0; i < nDim * mDim; i++)
		{
			if (abs(hostC[i] - C[i]) > 100.0f)
			{
				isEqual = false;
				cout << "Different value on index: " << i << "\n";
				cout << "Difference: " << abs(hostC[i] - C[i]) << "\n";
				cout << hostC[i] << " != " << C[i] << "\n";
				break;
			}
		}
		cout << "Equality: " << boolalpha << isEqual << "\n";
	}

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
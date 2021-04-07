/* Example of chaining kernel executions and using events to synchronize them.
* Hadamard product of vectors:
* A * A = B
* B * B = C
* C * A = D
* C * B = E
* D * E = F
*/

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

// Use opencl.hpp instead of cl2.hpp to make it clear that it supports all versions of OpenCL
// #include <CL/cl2.hpp>
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>

#define RAND_BASE 10
#define LENGTH 2048
#define VERBOSE false

using namespace std;

// Global variables for simplyfing code.
const cl_uint length = LENGTH;
cl_float vecA[LENGTH];
cl_float vecB[LENGTH];
cl_float vecC[LENGTH];
cl_float vecD[LENGTH];
cl_float vecE[LENGTH];
cl_float vecF[LENGTH];

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

void FillOrdered(cl_float* floatArray, cl_uint n, float start, float step)
{
	for (int i = 0; i < n; i++)
	{
		floatArray[i] = start + i * step;
	}
}

void FillRandom(cl_float* floatArray, cl_uint n, bool normalized = false)
{
	srand(2021);
	for (int i = 0; i < n; i++)
	{
		floatArray[i] = rand() / (float)RAND_MAX;
		if (!normalized)
		{
			floatArray[i] *= RAND_BASE;
		}
	}
}

void FillEmpty(cl_float* floatArray, cl_uint n)
{
	for (int i = 0; i < n; i++)
	{
		floatArray[i] = 0.0f;
	}
}

void PrintVector(const float (&vec)[LENGTH])
{
	for (int i = 0; i < LENGTH; i++)
	{
		cout << vec[i] << " ";
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

void Profile(cl::Event& clStartEvent, cl::Event& clFinishEvent)
{
	cl_ulong startTime = clStartEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong endTime = clFinishEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	cl_ulong elapsed = endTime - startTime;
	cout << "Time elapsed: " << elapsed << " ns\n";
}

void HadamardProductChain(cl::Device& device, cl::Context& context, cl::Program& program)
{
	cout << "\n\nHadamard product - chaining version:\n";

	size_t sizeVec = sizeof(vecA);

	cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeVec);
	cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeVec);
	cl::Buffer bufferC(context, CL_MEM_READ_WRITE, sizeVec);
	cl::Buffer bufferD(context, CL_MEM_READ_WRITE, sizeVec);
	cl::Buffer bufferE(context, CL_MEM_READ_WRITE, sizeVec);
	cl::Buffer bufferF(context, CL_MEM_READ_WRITE, sizeVec);

	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	cl::CommandQueue commandQueue(context, device, properties);

	commandQueue.enqueueWriteBuffer(bufferA, true, 0, sizeVec, (void*)vecA);

	//cl::Kernel kernel(program, "HadamardProduct");
	cl::Kernel kernels[5]{
		cl::Kernel(program, "HadamardProduct"),
		cl::Kernel(program, "HadamardProduct"),
		cl::Kernel(program, "HadamardProduct"),
		cl::Kernel(program, "HadamardProduct"),
		cl::Kernel(program, "HadamardProduct")
	};

	// A * A = B
	kernels[0].setArg(0, bufferA);
	kernels[0].setArg(1, bufferA);
	kernels[0].setArg(2, bufferB);
	kernels[0].setArg(3, sizeof(cl_uint), &length);

	// B * B = C
	kernels[1].setArg(0, bufferB);
	kernels[1].setArg(1, bufferB);
	kernels[1].setArg(2, bufferC);
	kernels[1].setArg(3, sizeof(cl_uint), &length);

	// C * A = D
	kernels[2].setArg(0, bufferC);
	kernels[2].setArg(1, bufferA);
	kernels[2].setArg(2, bufferD);
	kernels[2].setArg(3, sizeof(cl_uint), &length);

	// C * B = E
	kernels[3].setArg(0, bufferC);
	kernels[3].setArg(1, bufferB);
	kernels[3].setArg(2, bufferE);
	kernels[3].setArg(3, sizeof(cl_uint), &length);

	// D * E = F
	kernels[4].setArg(0, bufferD);
	kernels[4].setArg(1, bufferE);
	kernels[4].setArg(2, bufferF);
	kernels[4].setArg(3, sizeof(cl_uint), &length);

	cl::NDRange global(length);
	cl::NDRange local = kernels[0].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

	cl::Event startEvent;
	cl::Event finishEvent;
	commandQueue.enqueueNDRangeKernel(kernels[0], cl::NullRange, global, local, NULL, &startEvent);
	commandQueue.enqueueNDRangeKernel(kernels[1], cl::NullRange, global, local, NULL, NULL);
	commandQueue.enqueueNDRangeKernel(kernels[2], cl::NullRange, global, local, NULL, NULL);
	commandQueue.enqueueNDRangeKernel(kernels[3], cl::NullRange, global, local, NULL, NULL);
	commandQueue.enqueueNDRangeKernel(kernels[4], cl::NullRange, global, local, NULL, &finishEvent);

	commandQueue.finish();

	Profile(startEvent, finishEvent);

	// Reading results:
	commandQueue.enqueueReadBuffer(bufferA, true, 0, sizeVec, (void*)vecA);
	commandQueue.enqueueReadBuffer(bufferB, true, 0, sizeVec, (void*)vecB);
	commandQueue.enqueueReadBuffer(bufferC, true, 0, sizeVec, (void*)vecC);
	commandQueue.enqueueReadBuffer(bufferD, true, 0, sizeVec, (void*)vecD);
	commandQueue.enqueueReadBuffer(bufferF, true, 0, sizeVec, (void*)vecF);
	if (VERBOSE)
	{
		PrintVector(vecF);
	}
}

int Program(int argc, char* argv[])
{
	cl::Platform platform = FindOpenCLPlatform();

	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(), 0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, contextProperties);
	vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
	cl::Device device = devices[0];

	// Read source file
	ifstream sourceFile("HadamardProduct.cl");
	string kernelSource(
		istreambuf_iterator<char>(sourceFile),
		(istreambuf_iterator<char>()));
	cl::Program::Sources source{ kernelSource };
	cl::Program program = cl::Program(context, source);
	// Build binary version of program.
	program.build(device);

	cout << "\n";

	FillRandom(vecA, length);
	FillEmpty(vecB, length);
	FillEmpty(vecC, length);
	FillEmpty(vecD, length);
	FillEmpty(vecE, length);
	FillEmpty(vecF, length);

	// Printing matrices to test out.
	if (VERBOSE)
	{
		PrintVector(vecA);
	}

	HadamardProductChain(device, context, program);


	//// Main kernel program
	////KernelSgemmNaive(program, commandQueue, nDim, kDim, mDim, bufferA, bufferB, bufferC);
	////KernelSgemmComputeUnits(device, program, commandQueue, nDim, kDim, mDim, bufferA, bufferB, bufferC);
	////KernelSgemmPrivate(device, program, commandQueue, nDim, kDim, mDim, bufferA, bufferB, bufferC);
	//KernelSgemmLocal(device, program, commandQueue, nDim, kDim, mDim, bufferA, bufferB, bufferC);

	//// Read and check results
	//commandQueue.enqueueReadBuffer(bufferC, true, 0, sizeC, (void*)C);
	//if (PRINT_MATRICES)
	//{
	//	PrintMatrix(C, nDim, mDim);
	//}

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
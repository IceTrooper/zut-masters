#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#define CHECK_ERROR(err, DESC) \
if(err != CL_SUCCESS) \
{ \
	printf(DESC); \
	return err; \
}

#define CHECK_ERROR_ARG(err, DESC, arg) \
if(err != CL_SUCCESS || arg) \
{ \
	printf(DESC); \
	return err; \
}

/* This function helps to create informative messages in
 * case when OpenCL errors occur. It returns a string
 * representation for an OpenCL error code.
 * (E.g. "CL_DEVICE_NOT_FOUND" instead of just -1.)
 */
const char* TranslateOpenCLError(cl_int errorCode)
{
	switch (errorCode)
	{
	case CL_SUCCESS:                            return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
	case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
	case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
	case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
	case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
	case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
	case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
	case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
	case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
	case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
	case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
//    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
//    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

	default:
		return "UNKNOWN ERROR CODE";
	}
}

const char* kernelSource =
	"__kernel void Saxpy(const float a, __global const float* x, __global const float *y, __global float *z, const int N)"
	"{"
	"int gid = get_global_id(0);"
	"	if (gid < N)"
	"	{"
	"		z[gid] = a * x[gid] + y[gid];"
	"	}"
	"}";

bool CheckPreferredPlatformMatch(cl_platform_id platform, const char* preferredPlatform)
{
	size_t stringLength = 0;
	cl_int err = CL_SUCCESS;

	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &stringLength);
	if (err != CL_SUCCESS)
	{
		printf("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned an error!\n");
		return false;
	}

	char* platformName = (char*)malloc(stringLength * sizeof(char));
	if (platformName == NULL)
	{
		printf("Error: Cannot allocate platformName!\n");
		return false;
	}

	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, stringLength, platformName, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME returned an error!\n");
		free((void*)platformName);
		return false;
	}

	if (strstr(platformName, preferredPlatform) == 0)
	{
		free((void*)platformName);
		return false;
	}

	printf("Platform: %s\n", platformName);
	free((void*)platformName);
	return true;
}

cl_platform_id FindOpenCLPlatform()
{
	cl_uint numPlatforms = 0;
	cl_int err = CL_SUCCESS;

	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS)
	{
		printf("Error: clGetPlatformIDs() to get number of platforms returned an error!\n");
		return NULL;
	}

	printf("Number of available platforms: %u\n", numPlatforms);

	if (numPlatforms == 0)
	{
		printf("Error: No platforms found!\n");
		return NULL;
	}

	cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
	if (platforms == NULL)
	{
		printf("Error: Cannot allocate platforms!\n");
		return NULL;
	}

	err = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: clGetPlatformIDs() to get platforms returned an error!\n");
		free((void*)platforms);
		return NULL;
	}

	// Looking for Intel platform
	for (cl_uint i = 0; i < numPlatforms; i++)
	{
		cl_uint numDevices = 0;

		if (!CheckPreferredPlatformMatch(platforms[i], "Intel"))
		{
			continue;
		}

		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
		CHECK_ERROR(err, "Required device was not found on this platform.\n");

		// Getting device
		if (numDevices != 0)
		{
			printf("Required device was found.\n");
			printf("Number of available devices: %u\n", numDevices);

			cl_platform_id platform = platforms[i];
			free((void*)platforms);
			return platform;
		}
	}

	printf("Error: Required device was not found on any platform!\n");
	free((void*)platforms);
	return NULL;
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
	cl_int err = CL_SUCCESS;

	cl_platform_id platformId = FindOpenCLPlatform();
	if (platformId == NULL)
	{
		printf("Error: Failed to find OpenCL platform!\n");
		return 0;
	}

	// Specifies a list of context property names and their corresponding values.
	// Each property name is immediately followed by the corresponding desired value.
	// The list is terminated with 0. properties can be NULL
	// in which case the platform that is selected is implementation-defined.
	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformId, 0 };
	cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
	CHECK_ERROR_ARG(err, "Couldn't create a context, clCreateContextFromType() returned an error!\n", (context == NULL));

	cl_device_id device = NULL;
	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
	CHECK_ERROR(err, "Error: clGetContextInfo() to get list of devices returned an error!\n");

	const int N = 32;
	size_t nBytes = N * sizeof(float);
	const float inputA = 2.5f;
	float* hostInputX = (float*)malloc(nBytes);
	float* hostInputY = (float*)malloc(nBytes);
	float* hostOutZ = (float*)malloc(nBytes);
	FillOrdered(hostInputX, N, 1.0f, 1.0f);
	FillOrdered(hostInputY, N, 1.0f, 2.0f);
	FillEmpty(hostOutZ, N);

	cl_mem deviceInX = clCreateBuffer(context, CL_MEM_READ_ONLY, nBytes, NULL, &err);
	cl_mem deviceInY = clCreateBuffer(context, CL_MEM_READ_ONLY, nBytes, NULL, &err);
	cl_mem deviceOutZ = clCreateBuffer(context, CL_MEM_WRITE_ONLY, nBytes, NULL, &err);
	CHECK_ERROR(err, "Error: clCreateBuffer() returned an error!\n");

	//const cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	cl_command_queue commandQueue = clCreateCommandQueue(context, device, properties, &err);
	CHECK_ERROR_ARG(err, "Error: Couldn't create command queue, clCreateCommandQueue() returned an error!\n", (commandQueue == NULL));

	err = clEnqueueWriteBuffer(commandQueue, deviceInX, CL_TRUE, 0, nBytes, (void*)hostInputX, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(commandQueue, deviceInY, CL_TRUE, 0, nBytes, (void*)hostInputY, 0, NULL, NULL);
	CHECK_ERROR(err, "Error: clEnqueueWriteBuffer() returned an error!\n");

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
	CHECK_ERROR(err, "Error: clCreateProgramWithSource() returned an error!\n");

	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	CHECK_ERROR(err, "Error: clBuildProgram() for source program returned an error!\n");

	cl_kernel kernel = clCreateKernel(program, "Saxpy", &err);
	CHECK_ERROR(err, "Error: clCreateKernel() returned an error!\n");

	// Setting kernel arguments
	err = clSetKernelArg(kernel, 0, sizeof(float), &inputA);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &deviceInX);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &deviceInY);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &deviceOutZ);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &N);
	CHECK_ERROR(err, "Error: clSetKernelArg() returned an error. Cannot set kernel arguments!\n");

	int globalWorkSize = N;
	err = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
	CHECK_ERROR(err, "Error: Failed to run kernel!\n");

	err = clEnqueueReadBuffer(commandQueue, deviceOutZ, CL_TRUE, 0, nBytes, &hostOutZ[0], 0, NULL, NULL);
	CHECK_ERROR(err, "Error: clEnqueueReadBuffer() returned an error!\n");

	for (int i = 0; i < N; i++)
	{
		printf("%f ", hostOutZ[i]);
	}
	printf("\n");

	// Free allocated memory
	free((void*)hostInputX);
	free((void*)hostInputY);
	free((void*)hostOutZ);
	return 0;
}

int main(int argc, char* argv[])
{
	int errorCode = Program(argc, argv);
	printf("Returned code (%i): %s\n", errorCode, TranslateOpenCLError(errorCode));
	system("pause");
	return errorCode;
}
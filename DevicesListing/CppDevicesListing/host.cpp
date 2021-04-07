#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

// Use opencl.hpp instead of cl2.hpp to make it clear that it supports all versions of OpenCL
// #include <CL/cl2.hpp>
#include <CL/opencl.hpp>
#include <iostream>
#include <iomanip>

// Helper function to print device type according to cl_device_type variable
void printDeviceType(cl_device_type device_type)
{
    std::cout << "CL_DEVICE_TYPE: ";

    if (device_type == CL_DEVICE_TYPE_CUSTOM)
    {
        std::cout << "CL_DEVICE_TYPE_CUSTOM\n";
        // We can safely return the function because
        // if device type is custom there are no other combinations.
        return;
    }

    // We use help macro to print a separator
    // if there were any other types printed before.
#define PRINT_SEPARATOR(SHOULD)  \
    if(SHOULD == true)           \
    {                            \
        std::cout << " | ";       \
    }

    // Device type can be a combination of the below types
    // so we have to bitwise check all of the types.
    bool printed_before = false;
    if (device_type & CL_DEVICE_TYPE_CPU)
    {
        std::cout << "CL_DEVICE_TYPE_CPU";
        printed_before = true;
    }

    if (device_type & CL_DEVICE_TYPE_GPU)
    {
        PRINT_SEPARATOR(printed_before);
        std::cout << "CL_DEVICE_TYPE_GPU";
        printed_before = true;
    }

    if (device_type & CL_DEVICE_TYPE_ACCELERATOR)
    {
        PRINT_SEPARATOR(printed_before);
        std::cout << "CL_DEVICE_TYPE_ACCELERATOR";
        printed_before = true;
    }

    if (device_type & CL_DEVICE_TYPE_DEFAULT)
    {
        PRINT_SEPARATOR(printed_before);
        std::cout << "CL_DEVICE_TYPE_DEFAULT";
    }

    std::cout << std::endl;
}

int main()
{
	using namespace std;
	long long indentation_level = 0;

#define INDENT(LEVEL)   \
    cout                \
    << setw(4 * LEVEL)  \
    << " ";

	vector<cl::Platform> platforms;
	// Wraps clGetPlatformIDs()
	cl::Platform::get(&platforms);
	cout << "Number of available platforms: " << platforms.size() << "\n";

	cout << "Platform names:" << "\n";
	++indentation_level;
	for (auto& platform : platforms)
	{
        cout << "\n";
		INDENT(indentation_level);
		// Wrapper for clGetPlatformInfo()
		cout << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

		vector<cl::Device> devices;
		// Wraps clGetDeviceIDs()
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        // Iterate over every device on the platform and print useful capabilities information.

        INDENT(indentation_level);
        cout << "*** Detailed information for each device ***\n";

        ++indentation_level;
		for (auto& device : devices)
		{
            cout << "\n";
            // Helper macro
#define OCLBASIC_PRINT_PROPERTY(NAME)   \
            {                               \
                INDENT(indentation_level);  \
                cout << "" << #NAME << ": " << device.getInfo<NAME>() << "\n";     \
            }

            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_NAME);
            INDENT(indentation_level);
            printDeviceType(device.getInfo< CL_DEVICE_TYPE>());
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_AVAILABLE);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_VENDOR);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_PROFILE);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_VERSION);
            OCLBASIC_PRINT_PROPERTY(CL_DRIVER_VERSION);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_OPENCL_C_VERSION);

            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_MAX_COMPUTE_UNITS);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_MAX_CLOCK_FREQUENCY);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_MAX_WORK_GROUP_SIZE);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_ADDRESS_BITS);

            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_MEM_BASE_ADDR_ALIGN);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_GLOBAL_MEM_SIZE);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_LOCAL_MEM_SIZE);

            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_PROFILING_TIMER_RESOLUTION);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_IMAGE_SUPPORT);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_ERROR_CORRECTION_SUPPORT);
            // Deprecated in OpenCL 2.0
            //OCLBASIC_PRINT_PROPERTY(CL_DEVICE_HOST_UNIFIED_MEMORY);

            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_EXTENSIONS);

            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT);
            OCLBASIC_PRINT_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE);
		}
        --indentation_level;
	}
	--indentation_level;

	system("pause");
}
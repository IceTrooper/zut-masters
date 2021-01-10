#include <iostream>
#include <iomanip>
#include <cassert>

#include <CL/cl.h>

// Helper function to print device type according to cl_device_type variable
void printDeviceType(cl_device_type device_type)
{
    std::cout << "CL_DEVICE_TYPE: ";

    if (device_type == CL_DEVICE_TYPE_CUSTOM)
    {
        std::cout << "CL_DEVICE_TYPE_CUSTOM" << std::endl;
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

int main(int argc, const char** argv)
{
    // All stuff needed for sample is kept in this function body.
    // There is a couple of help macros; so they are also defined
    // directly inside this function and context dependent.

    using namespace std;
    long long indentation_level = 0;

#define INDENT(LEVEL)   \
    cout                \
    << setw(4 * LEVEL)  \
    << " ";

    // -----------------------------------------------------------------------
    // 1. Define error handling strategy.

    // The following variable stores return codes for all OpenCL calls.
    // In the code it is used with CAPSBASIC_CHECK_ERRORS macro defined next.
    cl_int err = CL_SUCCESS;

    // Error handling strategy for this sample is fairly simple -- just print
    // a message and terminate the application if something goes wrong.
#define CAPSBASIC_CHECK_ERRORS(ERR)        \
    if(ERR != CL_SUCCESS)                  \
    {                                      \
    cerr                                   \
    << "OpenCL error with code " << ERR    \
    << " happened in file " << __FILE__    \
    << " at line " << __LINE__             \
    << ". Exiting...\n";                   \
    exit(1);                               \
    }

    // -----------------------------------------------------------------------
    // 2. Query for all available OpenCL platforms on the system

    cl_uint num_of_platforms = 0;
    // get total number of available platforms:
    err = clGetPlatformIDs(0, 0, &num_of_platforms);
    CAPSBASIC_CHECK_ERRORS(err);
    cout << "Number of available platforms: " << num_of_platforms << endl;

    cl_platform_id* platforms = new cl_platform_id[num_of_platforms];
    // get IDs for all platforms:
    err = clGetPlatformIDs(num_of_platforms, platforms, 0);
    CAPSBASIC_CHECK_ERRORS(err);


    // -----------------------------------------------------------------------
    // 3. List all platforms and all devices under every platform.

    cout << "Platform names:\n";

    ++indentation_level;
    for (cl_uint i = 0; i < num_of_platforms; ++i)
    {

        // Get the length for the i-th platform name
        size_t platform_name_length = 0;
        err = clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_NAME,
            0,
            0,
            &platform_name_length
        );
        CAPSBASIC_CHECK_ERRORS(err);

        // Get the name itself for the i-th platform
        char* platform_name = new char[platform_name_length];
        err = clGetPlatformInfo(
            platforms[i],
            CL_PLATFORM_NAME,
            platform_name_length,
            platform_name,
            0
        );
        CAPSBASIC_CHECK_ERRORS(err);

        cout << "\n";
        INDENT(indentation_level);
        cout << "[" << i << "] " << platform_name << endl;
        delete[] platform_name;

        cl_platform_id platform = platforms[i];

        // Get number of all devices of all types for current platform.
        cl_uint num_of_devices = 0;
        err = clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_ALL,
            0,
            0,
            &num_of_devices
        );
        CAPSBASIC_CHECK_ERRORS(err);

        INDENT(indentation_level);
        cout << "Number of devices available: " << num_of_devices << endl;

        // Now get all devices under current platform.
        cl_device_id* devices = new cl_device_id[num_of_devices];
        err = clGetDeviceIDs(
            platform,
            CL_DEVICE_TYPE_ALL,
            num_of_devices,
            devices,
            0
        );
        CAPSBASIC_CHECK_ERRORS(err);

        // -----------------------------------------------------------------------
        // 4. Now get a piece of useful capabilities information for each device.

        INDENT(indentation_level);
        cout << "*** Detailed information for each device ***\n";

        ++indentation_level;
        // Iterate over all devices of the current platform.
        for (cl_uint device_index = 0; device_index < num_of_devices; ++device_index)
        {
            INDENT(indentation_level);
            cout << "[" << device_index << "]\n";

            cl_device_id device = devices[device_index];

            // To enumerate capabilities information, use two help
            // macros: one to print string information and another one to
            // print numeric information. Both these macros use clGetDeviceInfo
            // to retrieve required caps, and defined below:

#define OCLBASIC_PRINT_TEXT_PROPERTY(NAME)                       \
            {                                                    \
            /* When we query for string properties, first we */  \
            /* need to get string length:                    */  \
            size_t property_length = 0;                          \
            err = clGetDeviceInfo(                               \
            device,                                              \
            NAME,                                                \
            0,                                                   \
            0,                                                   \
            &property_length                                     \
            );                                                   \
            CAPSBASIC_CHECK_ERRORS(err);                         \
            /* Then allocate buffer. No need to add 1 symbol */  \
            /* to store terminating zero; OpenCL takes care  */  \
            /* about it:                                     */  \
            char* property_value = new char[property_length];    \
            err = clGetDeviceInfo(                               \
            device,                                              \
            NAME,                                                \
            property_length,                                     \
            property_value,                                      \
            0                                                    \
            );                                                   \
            CAPSBASIC_CHECK_ERRORS(err);                         \
            INDENT(indentation_level)                            \
            cout                                                 \
            << "" << #NAME << ": "                               \
            << property_value << endl;                           \
            delete [] property_value;                            \
            }

#define OCLBASIC_PRINT_NUMERIC_PROPERTY(NAME, TYPE)              \
            {                                                    \
            TYPE property_value;                                 \
            size_t property_length = 0;                          \
            err = clGetDeviceInfo(                               \
            device,                                              \
            NAME,                                                \
            sizeof(property_value),                              \
            &property_value,                                     \
            &property_length                                     \
            );                                                   \
            assert(property_length == sizeof(property_value));   \
            CAPSBASIC_CHECK_ERRORS(err);                         \
            INDENT(indentation_level)                            \
            cout                                                 \
            << #NAME << ": "                                     \
            << property_value << endl;                           \
            }

            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_NAME);
            cl_device_type device_type;
            clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
            INDENT(indentation_level);
            printDeviceType(device_type);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_AVAILABLE, cl_bool);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_VENDOR);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_PROFILE);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_VERSION);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DRIVER_VERSION);
            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_OPENCL_C_VERSION);

            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_WORK_GROUP_SIZE, size_t);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_ADDRESS_BITS, cl_uint);

            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong);

            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PROFILING_TIMER_RESOLUTION, size_t);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_IMAGE_SUPPORT, cl_bool);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_HOST_UNIFIED_MEMORY, cl_bool);

            OCLBASIC_PRINT_TEXT_PROPERTY(CL_DEVICE_EXTENSIONS);

            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, cl_uint);
            OCLBASIC_PRINT_NUMERIC_PROPERTY(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, cl_uint);

        }
        --indentation_level;
        delete[] devices;

    }
    --indentation_level;

    // -----------------------------------------------------------------------
    // Final clean up

    delete[] platforms;

    system("pause");
    return 0;
}

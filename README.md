# zut-masters
Master's degree in Computer Science on The West Pomeranian University of Technology in Szczecin.

# Contents
Repository contains code examples for master thesis about OpenCL running on Intel GPU.

List of examples:
1. DeviceListing - print informations about OpenCL on devices.
2. SAXPY - basic usage of kernels in OpenCL.
3. SGEMM - optimization techniques for data parallel programs.
4. Hadamard product - chaining kernels and using events for synchronization between them.
5. Parallelism - data and task based parallelism.
6. Images - basic operations on images and samplers.
	- Scale image with GPU's sampler
	- Apply filters to images
	- Draw Sierpinski Triangle
7. PyOpenCL - example of using sgemm kernel in Python language (host side).

Features of OpenCL shown in those examples:
1. Reading kernels from string and files (C/C++ versions).
2. Private, local, global memory usage.
3. Profiling events.
4. Chaining kernels and synchronization events.
5. In-Order and Out-of-Order execution.
6. Data and task based parallelism.
7. Offline/Online compilation.
8. Using images and samplers.
9. Building kernel programs for OpenCL 2.0 (-cl-std=CL2.0 flag).
10. PyOpenCL - using OpenCL with Python language.

## DeviceListing
This project shows how to get all platforms/devices and their informations about OpenCL support. There is C++ and C version of the same project.

### Resources
- [Platform and Device Capabilities Viewer](https://software.intel.com/content/dam/develop/public/us/en/downloads/intel_ocl_caps_basic.zip)

## SAXPY
Computing equation of z=a*x+y known as SAXPY (Single precision AX Plus Y).

There are 4 versions of this program:
1. C with kernel source inside string
2. C with kernel source inside file
3. C++ with kernel source inside string
4. C++ with kernel source inside file

### Resources
- "OpenCL Programming
by Example" (p. 26)
- "OpenCL Akceleracja GPU w praktyce" (p. 111, 120)

## SGEMM
Example of matrices multiplication C (n x m) = A (n x k) * B (k x m) known as SGEMM operation.

There are 4 different kernel versions, from the most basic to the most optimized and fastest:
1. Simple SGEMM is naive implementation of sequentional based SGEMM operation.
2. Increase amount of work per work item. Now work item compute results of entire row of C. It's slower than first example, but it's more promising.
3. Copy entire row of A from global into private work item memory.
4. Copy columns of B from global into local work group memory.

### Notes
You can't pass pointer of pointers to kernel so you need to [reduce 2d matrix into 1d array of values](https://stackoverflow.com/questions/35442327/2d-array-as-opencl-kernel-argument).

### Resources
- "OpenCL Programming Guide" (p. 499-513)

## Hadamard Product
The program computes Hadamard product showing how to use simple kernel chaining and events to synchronize between their calls.

There are 2 versions of this program:
1. Simple kernel chaining.
2. Out-of-order command queue with events to synchronize between kernel calls.

### Notes
- [You can't use profiling events](https://community.intel.com/t5/OpenCL-for-CPU/Out-of-Order-Queues-do-they-work-Enqueued-Barriers-with-Events/td-p/1182479) in Out Of Order command queue. In this example time is measured with chrono on host side. [Intel example](https://github.com/intel/compute-samples/tree/master/compute_samples/applications/commands_aggregation)
- Some devices don't support OOQ. When you want to mimic out-of-order execution you need to make multiple command queues and synchronize operations between them.

### Resources
- [OpenCL: A Hands-on Introduction; Tim Mattson, Alice Koniges; .pdf presentation](https://www.nersc.gov/assets/pubs_presos/MattsonTutorialSC14.pdf)
- [A progression of OpenCL exercises; Tim Mattson](https://indico.cern.ch/event/138427/sessions/11396/attachments/116551/165426/OpenCL-exercises.pdf)
- [OpenCL™ Out-of-Order Queue on Intel® Processor Graphics ](https://software.intel.com/content/www/us/en/develop/articles/opencl-out-of-order-queue-on-intel-processor-graphics.html)

## Parallelism
This example shows data and task based parallelism. Three simple operations on arrays (adding, substracting, multiplying) are defined in kernels. There is one kernel needed in data parallel example and 3 different kernels for each operation in task parallel example.

### Notes
- **cl::CommandQueue::enqueueTask** is equivalent to calling **cl::CommandQueue::enqueueNDRangeKernel** with *work_dim = 1, global = NULLRange, global[0] set to 1 and local[0] set to 1*; [reference](https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.2.pdf)

### Resources
- [The OpenCL Programming Book
FREE HTML version](https://us.fixstars.com/products/opencl/book/OpenCLProgrammingBook/calling-the-kernel/)

## Images
This solution contains 3 examples:
1. ImageScaling - image scaling functionality using GPU's sampler. Kernel code is based on [Quick Getting Started Guide for creating Intel® OpenCL SDK application in Microsoft Visual Studio*](https://software.intel.com/content/www/us/en/develop/articles/quick-getting-started-guide-for-intel-opencl-sdk-integration-in-intel-system-studio-2019.html).
2. ImageFilters - applying various filters (average, sharpen, blur) to images.
3. SierpinskiTriangle - drawing Sierpinski triangle to the image using OpenCL 2.0 features. This example shows nested parallelism and the use case of enqueue_kernel. It's based on [Intel's Sierpinski Carpet example](https://software.intel.com/content/www/us/en/develop/articles/sierpinski-carpet-in-opencl-20.html).

### Notes
- In those examples we use [CImg](https://cimg.eu/) library to read/write images on host side. You don't need to install it, because every project in this solution contains attached CImg.h file which is in 'common' directory. This library supports natively most basics formats (i.e. PPM). In the second example - ImageFilters we use .png format just to show how it works. If you want to run it you need to have [ImageMagick](https://imagemagick.org/index.php) installed on PC then CImg will find it and would be able to support more formats (PNG, JPG, etc.)
- [When to use Images and Buffers](https://software.intel.com/content/www/us/en/develop/documentation/iocl-opg/top/check-list-for-opencl-optimizations/using-buffers-and-images-appropriately.html)
- [You can't use CL_RGB channel order in cl_image_format with unnormalized image_channel_data_type](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/cl_image_format.html). Then in each example we use CImg to add 4th alpha channel to images with 255 value.
- [Quicksort OpenCL 2.0: Nested Parallelism example](https://software.intel.com/content/www/us/en/develop/articles/gpu-quicksort-in-opencl-20-using-nested-parallelism-and-work-group-scan-functions.html)

### CImg problems when using with OpenCL
It's good to know that CImg uses planar format (R1R2R3...G1G2G3...B1B2B3), when OpenCL's image2d_t uses interleaved instead (R1G1B1R2G2B2R3G3B3...).
More informations:
- [Simple OpenCL kernel produces vertical bars instead of solid color in output image. Why?](https://stackoverflow.com/questions/12921288/simple-opencl-kernel-produces-vertical-bars-instead-of-solid-color-in-output-ima)
- [CImg - how to convert interleaved raw data?](https://stackoverflow.com/questions/48231065/cimg-how-to-convert-interleaved-raw-data)
- [CImg does not store pixels in the interleaved format ](https://www.codefull.org/2014/11/cimg-does-not-store-pixels-in-the-interleaved-format/)

## PyOpenCL
This example shows how to use Python language for host code in OpenCL. For this purpose [PyOpenCL](https://github.com/inducer/pyopencl) package is used. For kernel code final, most optimized version of sgemm is used. It's shown:
- how to copy buffers,
- how to pass dynamic array for local memory in kernel
- how to pass scalar values
- how to use exception handling

### Notes
- There are more PyOpenCL [examples in official package repository](https://github.com/inducer/pyopencl/tree/main/examples).

# Additional informations
## Useful resources
- [Intel Educational Resources](https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk/training.html#codesamples)
- [OpenCL C++ Bindings Documentation](https://github.khronos.org/OpenCL-CLHPP/index.html)
- [Optimization Techniques for Device (DSP) Code](https://downloads.ti.com/mctools/esd/docs/opencl/optimization/dsp_code.html)
- [Check-list for OpenCL Optimizations](https://software.intel.com/content/www/us/en/develop/documentation/iocl-opg/top/check-list-for-opencl-optimizations/mapping-memory-objects.html)
- "OpenCL Akceleracja GPU w praktyce" Marek Sawerwain, 2014
- "OpenCL Programming Guide" by Aaftab Munshi, Benedict R. Gaster, Timothy G. Mattson, James Fung, Dan Ginsburg, 2012
- "OpenCL Programming by Example" by Ravishekhar Banger, Koushik Bhattacharyya, 2013

## More ideas for parallel computing programs
Computing Pi number, Bitonic sort, Drawing gradient, Image filters (mean, median, gaussian, sobel), Reduction, histogram, convolve, game of life.

## Interesting facts
- [Intel® Graphics device is much faster for floating-point add, sub, mul and so on in compare to the int type.](https://software.intel.com/content/www/us/en/develop/documentation/iocl-opg/top/check-list-for-opencl-optimizations/using-floating-point-for-calculations.html)
- Out-of-Order; [Turning on profiling for a given command queue prevents concurrent execution](https://github.com/intel/compute-samples/tree/master/compute_samples/applications/commands_aggregation#limitations)
- [OpenCL 2.0 allows a kernel to independently enqueue to the same device, without host interaction.](https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/enqueue_kernel.html)
- [In OpenCL 2.0 you can use Read-Write Images](https://software.intel.com/content/www/us/en/develop/articles/using-opencl-20-read-write-images.html)
- [Arguments to __kernel functions in a program cannot be declared as a pointer to a pointer(s).](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/restrictions.html)
- [OpenCL: How to check for build errors using the C++ wrapper](https://stackoverflow.com/questions/34662333/opencl-how-to-check-for-build-errors-using-the-c-wrapper)

## Assets
- intel.ppm - [Slejven Djurakovic; unsplash.com](https://unsplash.com/photos/0uXzoEzYZ4I)
- colorful.ppm - [USGS; unsplash.com](https://unsplash.com/photos/vOQV-8SFwZk)

## Author
[IceTrooper](https://github.com/IceTrooper/)
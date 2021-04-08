# zut-masters
Master's degree in Computer Science on The West Pomeranian University of Technology in Szczecin

# Contents
Repository contains code examples for master thesis about OpenCL.

List of examples:
1. DeviceListing - print informations about OpenCL on devices.
2. SAXPY - basic usage of kernels in OpenCL.
3. SGEMM - optimization techniques for data parallel programs.
4. Hadamard product - chaining kernels and using events for synchronization between them.
5. Parallelism - data and task based parallelism.
6. Images - basic operations on images and samplers.
7. Sierpinski triangle
8. PyOpenCL

Features of OpenCL shown in those examples:
1. Reading kernels from string and files (C/C++ versions).
2. Private, local, global memory usage.
3. Profiling events.
4. Chaining kernels and synchronization events.
5. In-Order and Out-of-Order execution.
6. Data and task based parallelism.
7. Offline/Online compilation.
8. Using images and samplers.
9. PyOpenCL - using OpenCL with Python language.

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
- You can't use profiling events in Out Of Order command queue. In this example time is measured with chrono on host side.
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

## PyOpenCL

# Additional informations
## Useful resources
[Intel Educational Resources](https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk/training.html#codesamples)

[OpenCL C++ Bindings Documentation](https://github.khronos.org/OpenCL-CLHPP/index.html)

"OpenCL Akceleracja GPU w praktyce" Marek Sawerwain, 2014

"OpenCL
Programming Guide" by Aaftab Munshi, Benedict R. Gaster, Timothy G. Mattson, James Fung, Dan Ginsburg, 2012

"OpenCL Programming
by Example" by Ravishekhar Banger,
Koushik Bhattacharyya, 2013

More ideas for parallel computing programs:
Computing Pi number, Bitonic sort, Drawing gradient, Image filters (mean, median, gaussian, sobel)

## Author
[IceTrooper](https://github.com/IceTrooper/)
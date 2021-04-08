__kernel void TaskParallelAdd(__global float* A, __global float* B, __global float* C, const uint rowsCount)
{
	int id = 0;
	for(int i = 0; i < rowsCount; i++)
	{
		id = i*3;
		C[id] = A[id] + B[id];
	}
}

__kernel void TaskParallelSub(__global float* A, __global float* B, __global float* C, const uint rowsCount)
{
	int id = 0;
	for(int i = 0; i < rowsCount; i++)
	{
		id = i*3;
		C[id+1] = A[id+1] + B[id+1];
	}
}

__kernel void TaskParallelMul(__global float* A, __global float* B, __global float* C, const uint rowsCount)
{
	int id = 0;
	for(int i = 0; i < rowsCount; i++)
	{
		id = i*3;
		C[id+2] = A[id+2] + B[id+2];
	}
}
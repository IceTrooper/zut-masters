__kernel void DataParallel(__global float* A, __global float* B, __global float* C, const uint rowsCount)
{
	int rowId = get_global_id(0);
	if(rowId < rowsCount)
	{
		rowId *= 3;
		C[rowId] = A[rowId] + B[rowId];
		C[rowId+1] = A[rowId+1] - B[rowId+1];
		C[rowId+2] = A[rowId+2] * B[rowId+2];
	}
}
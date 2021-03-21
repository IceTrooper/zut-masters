__kernel void Saxpy(const float a, __global const float* x, __global const float* y, __global float* z, const int N)
{
	int gid = get_global_id(0);
	if (gid < N)
	{
		z[gid] = a * x[gid] + y[gid];
	}
}
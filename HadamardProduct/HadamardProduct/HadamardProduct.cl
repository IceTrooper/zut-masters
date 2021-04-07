__kernel void HadamardProduct(__global const float* aVec, __global const float* bVec, __global float* cVec,
	const unsigned int length)
{
	int i = get_global_id(0);
	if(i < length)
	{
		cVec[i] = aVec[i] * bVec[i];
	}
}
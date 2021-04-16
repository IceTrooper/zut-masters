// Matrix multiplication
__kernel void Sgemm(const uint nDim, const uint kDim, const uint mDim,
    const __global float* A,  const __global float* B, __global float* C,
    __local float* localB)
{
    int i = get_global_id(0);
    int k, j;
    float acc;

    // This should be the same size value as kDim
    float privateA[2400];

    int localK = get_local_id(0);
    int localM = get_local_size(0);

    if(i < nDim)
    {
        // Copying from global to private memory.
        for(k = 0; k < kDim; k++)
        {
            privateA[k] = A[i*kDim + k];
        }
        
        for(j = 0; j < mDim; j++)
        {
            // Copying from global to local memory.
            for(k = localK; k < kDim; k+=localM)
            {
                localB[k] = B[k * mDim + j];
            }
        
            // Wait for all work items in group.
            barrier(CLK_LOCAL_MEM_FENCE);

            acc = 0.0f;        
            for(k = 0; k < kDim; k++)
            {
                // Now we're getting B values from faster local memory and A values from fastest private memory.
                acc += privateA[k] * localB[k];
            }
        
            C[i*mDim + j] = acc;
        }
    }
}

__kernel void Printing(const uint nDim)
{
    printf("%s", "halko");
}
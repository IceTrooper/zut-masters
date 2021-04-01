// Multiplication of A (n x k) by B (k x m)
__kernel void Sgemm(const int nDim, const int kDim, const int mDim, const __global float* A,  const __global float* B, __global float* C)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k;
    float acc = 0.0f;

    if(i < nDim && j < mDim)
    {
        for(k = 0; k < kDim; k++)
        {
            acc += A[i*nDim + k] * B[k*kDim + j];
        }
        C[i*nDim + j] = acc;
    }
}
// Multiplication of A (n x k) by B (k x m)
// rows (-) x cols (|)

__kernel void Sgemm_naive(const uint nDim, const uint kDim, const uint mDim, const __global float* A,  const __global float* B, __global float* C)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k;
    float acc = 0.0f;

    if(i < nDim && j < mDim)
    {
        for(k = 0; k < kDim; k++)
        {
            acc += A[i*kDim + k] * B[k*mDim + j];
        }
        C[i*mDim + j] = acc;
    }
}

__kernel void Sgemm_compute_units(const uint nDim, const uint kDim, const uint mDim, const __global float* A,  const __global float* B, __global float* C)
{
    int i = get_global_id(0);
    int k, j;
    float acc = 0.0f;

    if(i < nDim)
    {
        for(j=0; j<mDim; j++)
        {
            acc = 0.0f;
            for(k = 0; k < kDim; k++)
            {
                acc += A[i*kDim + k] * B[k*mDim + j];
            }
            C[i*mDim + j] = acc;
        }
    }
}

__kernel void Sgemm_private(const uint nDim, const uint kDim, const uint mDim, const __global float* A,  const __global float* B, __global float* C)
{
    int i = get_global_id(0);
    int k, j;
    float acc = 0.0f;

    if(i < nDim)
    {
        for(j=0; j<mDim; j++)
        {
            acc = 0.0f;
            for(k = 0; k < kDim; k++)
            {
                acc += A[i*kDim + k] * B[k*mDim + j];
            }
            C[i*mDim + j] = acc;
        }
    }
}
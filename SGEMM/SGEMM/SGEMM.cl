// Multiplication of A (n x k) by B (k x m)
// rows (-) x cols (|)

#include "host.h"

__kernel void Sgemm_naive(const uint nDim, const uint kDim, const uint mDim,
    const __global float* A,  const __global float* B, __global float* C)
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

// Each work item computes entire row of C.
__kernel void Sgemm_compute_units(const uint nDim, const uint kDim, const uint mDim,
    const __global float* A,  const __global float* B, __global float* C)
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

// Copy entire row into private (fastest) work item memory.
__kernel void Sgemm_private(const uint nDim, const uint kDim, const uint mDim,
    const __global float* A,  const __global float* B, __global float* C)
{
    int i = get_global_id(0);
    int k, j;
    float acc = 0.0f;

    float privateA[K_DIM];

    if(i < nDim)
    {
        // Copying from global to private memory.
        for(k = 0; k < kDim; k++)
        {
            privateA[k] = A[i*kDim + k];
        }

        for(j=0; j<mDim; j++)
        {
            acc = 0.0f;
            for(k = 0; k < kDim; k++)
            {
                // Now we're getting A values from faster private memory.
                acc += privateA[i*kDim + k] * B[k*mDim + j];
            }
            C[i*mDim + j] = acc;
        }
    }
}

// Copy entire row into private (fastest) work item memory.
__kernel void Sgemm_local(const uint nDim, const uint kDim, const uint mDim,
    const __global float* A,  const __global float* B, __global float* C,
    __local float* localB)
{
    int i = get_global_id(0);
    int k, j;
    float acc = 0.0f;

    float privateA[K_DIM];

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
            for(k = localK; k < localM; k+=localM)
            {
                localB[k] = B[k * kDim + j];
            }
        }

        for(j=0; j<mDim; j++)
        {
            acc = 0.0f;
            for(k = 0; k < kDim; k++)
            {
                // Now we're getting A values from faster private memory.
                acc += privateA[i*kDim + k] * B[k*mDim + j];
            }
            C[i*mDim + j] = acc;
        }
    }
}
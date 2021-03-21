// Multiplication of A (m x k) by B (k x n)
__kernel void Sgemm(const int M, const int K, const int N, const __global float* A,  const __global float* B, __global float* C, const int threadBlockSize) {
        
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: threadBlockSize)
    const int col = get_local_id(1); // Local col ID (max: threadBlockSize)
    const int globalRow = threadBlockSize * get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = threadBlockSize * get_group_id(1) + col; // Col ID of C (0..N)
     
    // Local memory to fit a tile of threadBlockSize*threadBlockSize elements of A and B
    __local float Asub[threadBlockSize][threadBlockSize];
    __local float Bsub[threadBlockSize][threadBlockSize];
     
    // Initialise the accumulation register
    float acc = 0.0f;
        
    // Loop over all tiles
    const int numTiles = K/threadBlockSize;
    for (int t=0; t<numTiles; t++) {
     
        // Load one tile of A and B into local memory
        const int tiledRow = threadBlockSize*t + row;
        const int tiledCol = threadBlockSize*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];
     
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
     
        // Perform the computation for a single tile
        for (int k=0; k<threadBlockSize; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
     
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
     
    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}
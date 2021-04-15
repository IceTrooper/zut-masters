//const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__kernel void Triangle(__write_only image2d_t resultImage, int imageSize, int xOffset, int yOffset,
                       int level, int maxLevel)
{
    int size = imageSize / pow(2.0f, level);
    int offset = xOffset + imageSize * yOffset;

    float4 colorRed = (float4)(1.0f, 0.0f, 0.0f, 1.0f);
    
    for(int i = 0; i < size; i++)
    {
        write_imagef(resultImage, (int2)(xOffset + i, yOffset + (size-1)), colorRed); // HORIZONTAL
        write_imagef(resultImage, (int2)(xOffset, yOffset + i), colorRed); // VERTICAL
        write_imagef(resultImage, (int2)(xOffset + i, yOffset + i), colorRed); // DIAGONAL
    }
    
    if(level < maxLevel)
    {
        queue_t q = get_default_queue();
        int newLevel = level + 1;
        int halfSize = size / 2;

        enqueue_kernel(q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(1), ^{Triangle(resultImage, imageSize, xOffset, yOffset, newLevel, maxLevel);});
        enqueue_kernel(q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(1), ^{Triangle(resultImage, imageSize, xOffset, yOffset + halfSize, newLevel, maxLevel);});
        enqueue_kernel(q, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(1), ^{Triangle(resultImage, imageSize, xOffset + halfSize, yOffset + halfSize, newLevel, maxLevel);});
    }
}
//const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__kernel void Triangle(__write_only image2d_t resultImage, int imageSize, int xOffset, int yOffset,
                       int level, int maxLevel)
{
    int size = imageSize / pow(2.0f, level);
    int offset = xOffset + imageSize * yOffset;

    float4 colorRed = (float4)(1.0f, 0.0f, 0.0f, 1.0f);
    
    for(int i = 0; i < imageSize; i++)
    {
        write_imagef(resultImage, (int2)(xOffset + i, yOffset + (size-1)), colorRed); // HORIZONTAL
        write_imagef(resultImage, (int2)(xOffset, yOffset + i), colorRed); // VERTICAL
        write_imagef(resultImage, (int2)(xOffset + i, yOffset + i), colorRed); // DIAGONAL
    }
    
    
    //write_imagef(resultImage, (int2)(0, 0), colorRed);
    if(level < maxLevel)
    {
        queue_t q = get_default_queue();
        int newLevel = level + 1;
        //const size_t  grid[1] = {1};
        //enqueue_kernel(q, 0, ndrange_1D(1), ^{Triangle(resultImage, imageSize, xOffset, yOffset, newLevel, maxLevel);});
    }

    //const size_t  grid[2] = {one_third, one_third};
    //enqueue_kernel(q, 0, ndrange_2D(grid), ^{ sierpinski (src, width, x+offsetx, y+offsety); });

    /*int2 coordinate = (int2)(get_global_id(0), get_global_id(1));

    int halfSize = windowSize / 2;
    float4 computedPixel = 0.0f;
    float4 pixel;
    int wx, wy;

    for(wx = -halfSize; wx <= halfSize; wx++)
    {
        for(wy = -halfSize; wy <= halfSize; wy++)
        {
            pixel = read_imagef(sourceImage, sampler, coordinate + (int2)(wx, wy));
            computedPixel += filter[(wx + halfSize) * windowSize + (wy + halfSize)] * pixel;
        }
    }

    write_imagef(destinationImage, coordinate, computedPixel / (windowSize * windowSize));*/
}
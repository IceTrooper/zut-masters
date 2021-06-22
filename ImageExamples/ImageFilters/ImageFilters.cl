const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
__kernel void Filter(__read_only image2d_t sourceImage, __write_only image2d_t destinationImage,
                            __constant float* filter, int windowSize)
{
    int2 coordinate = (int2)(get_global_id(0), get_global_id(1));

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

    write_imagef(destinationImage, coordinate, computedPixel / (windowSize * windowSize));
}
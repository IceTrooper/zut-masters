// https://software.intel.com/content/www/us/en/develop/articles/quick-getting-started-guide-for-intel-opencl-sdk-integration-in-intel-system-studio-2019.html
const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
__kernel void ImageScaling(__read_only image2d_t sourceImage, __write_only image2d_t destinationImage,
                            const float widthNormScalar, const float heightNormScalar)
{
    int2 coordinate = (int2)(get_global_id(0), get_global_id(1));
    float2 normalizedCoordinate = convert_float2(coordinate) * (float2)(widthNormScalar, heightNormScalar);
    float4 pixel = read_imagef(sourceImage, sampler, normalizedCoordinate);
    write_imagef(destinationImage, coordinate, pixel);
}
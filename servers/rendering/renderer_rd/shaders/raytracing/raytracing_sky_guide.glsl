#[compute]

#version 450

#extension GL_EXT_samplerless_texture_functions : require

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// DLSS Ray Reconstruction's diffuse albedo where the camera rays missed: the background as drawn (sky shader, its
// clouds and stars, the fog over it), so the denoiser passes it through. The miss shader only has the radiance octmap,
// blurred and without what a sky shader leaves out of its cubemap pass: Ray Reconstruction took the stars for lighting
// noise and smeared them as the camera turned.

layout(set = 0, binding = 0) uniform texture2D color_texture;
layout(r32f, set = 0, binding = 1) uniform restrict readonly image2D depth_image;
layout(rgba16f, set = 0, binding = 2) uniform restrict writeonly image2D diffuse_albedo_image;

layout(push_constant, std430) uniform Params {
	ivec2 size;
	uint pad[2];
}
params;

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.size))) {
		return;
	}
	if (imageLoad(depth_image, pixel).r != 0.0) {
		return;
	}
	// The same floor as DLSSRR_encodeDiffuseAlbedo().
	vec3 color = texelFetch(color_texture, pixel, 0).rgb;
	imageStore(diffuse_albedo_image, pixel, vec4(max(color, vec3(0.001)), 1.0));
}

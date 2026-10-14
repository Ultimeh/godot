#[compute]

#version 450

#extension GL_EXT_samplerless_texture_functions : require

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// DLSS Ray Reconstruction is a denoiser: a star smaller than an internal pixel, which every jittered frame catches or
// misses, looks like noise to it, and it smears it wherever it cannot lean on its history (behind clouds that move). The
// background has no noise: the sky shader draws it again at the output resolution, unjittered, and it replaces Ray
// Reconstruction's output where the internal pixels around are all background the transparent pass left alone. Near an
// edge, or under a particle, the output stays the denoiser's.

layout(set = 0, binding = 0) uniform texture2D background_texture; // Output resolution.
layout(r32f, set = 0, binding = 1) uniform restrict readonly image2D depth_image; // Internal resolution; 0 where the camera rays missed.
layout(set = 0, binding = 2) uniform texture2D color_texture; // Internal resolution, after the transparent pass.
layout(set = 0, binding = 3) uniform texture2D color_before_transparency_texture; // Internal resolution.
layout(rgba16f, set = 0, binding = 4) uniform restrict image2D output_image; // Ray Reconstruction's output.

layout(push_constant, std430) uniform Params {
	ivec2 output_size;
	ivec2 internal_size;
}
params;

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.output_size))) {
		return;
	}

	// The internal pixels this output pixel is made of, give or take the jitter (half an internal pixel).
	ivec2 center = ivec2((vec2(pixel) + 0.5) * vec2(params.internal_size) / vec2(params.output_size));
	for (int y = -1; y <= 1; y++) {
		for (int x = -1; x <= 1; x++) {
			ivec2 p = clamp(center + ivec2(x, y), ivec2(0), params.internal_size - 1);
			if (imageLoad(depth_image, p).r != 0.0) {
				return;
			}
			if (texelFetch(color_texture, p, 0) != texelFetch(color_before_transparency_texture, p, 0)) {
				return;
			}
		}
	}

	vec4 color = imageLoad(output_image, pixel);
	color.rgb = texelFetch(background_texture, pixel, 0).rgb;
	imageStore(output_image, pixel, color);
}

#[compute]

#version 450

#extension GL_EXT_samplerless_texture_functions : require

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// The path traced color into the scene's color buffer, as the raster's opaque pass leaves it
// (RenderForwardClustered::_render_scene()). Where the camera rays missed (depth at the far plane) is the background:
// with BG_CANVAS and BG_KEEP the pixels already in the color buffer stay (the raster loads them), else the miss color
// with the alpha of the raster's clear color, 0, which a transparent viewport shows through.

layout(set = 0, binding = 0) uniform texture2D source_texture;
layout(r32f, set = 0, binding = 1) uniform restrict readonly image2D depth_image;
layout(rgba16f, set = 0, binding = 2) uniform restrict writeonly image2D dest_image;

layout(push_constant, std430) uniform Params {
	ivec2 size;
	uint keep_background;
	uint pad;
}
params;

void main() {
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	if (any(greaterThanEqual(pixel, params.size))) {
		return;
	}

	vec4 color = texelFetch(source_texture, pixel, 0);
	if (imageLoad(depth_image, pixel).r == 0.0) {
		if (params.keep_background != 0u) {
			return;
		}
		color.a = 0.0;
	}

	// As the copy this replaces (CopyEffects::copy_to_rect() with p_sanitize_inf_nan).
	color = mix(color, vec4(100.0, 100.0, 100.0, 1.0), isinf(color));
	color = mix(color, vec4(100.0, 100.0, 100.0, 1.0), isnan(color));

	imageStore(dest_image, pixel, color);
}

#[compute]

#version 450

#VERSION_DEFINES

// The sky's sampling distribution for next event estimation of the Environment (raytracing_sky_sampling_inc.glsl):
// MODE_LUMINANCE reads the radiance octmap's sharpest level into the luminance of every texel, MODE_CDF sums them into
// the rows' and the marginal cumulative distributions.

#include "../oct_inc.glsl"
#include "raytracing_sky_sampling_inc.glsl"

#ifdef MODE_CDF
layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in; // RT_SKY_SAMPLING_SIZE rows.
#else
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
#endif

layout(set = 0, binding = 0, std430) restrict buffer SkySampling {
	float sky_sampling[];
};

#ifndef MODE_CDF
#ifdef USE_RADIANCE_OCTMAP_ARRAY
layout(set = 0, binding = 1) uniform texture2DArray radiance_octmap;
#else
layout(set = 0, binding = 1) uniform texture2D radiance_octmap;
#endif
layout(set = 0, binding = 2) uniform sampler radiance_sampler;

layout(push_constant, std430) uniform Params {
	vec2 border_size; // As SceneData's radiance_border_size for vec3_to_oct_with_border().
	uint pad[2];
}
params;
#endif

#ifdef MODE_CDF
shared float row_sums[RT_SKY_SAMPLING_SIZE];
shared float average;
shared bool above_average;
#endif

void main() {
#ifdef MODE_CDF
	const uint y = gl_LocalInvocationID.x;
	const uint row = y * RT_SKY_SAMPLING_SIZE;

	// MIS compensation (Karlik et al. 2019, "MIS Compensation: Optimizing Sampling Techniques in Multiple Importance
	// Sampling"; PBRT v4's ImageInfiniteLight compensatedDistribution): BRDF sampling already finds the sky's average,
	// so next event estimation samples only what is brighter than it. A uniform sky keeps its plain distribution.
	float sum = 0.0;
	for (uint x = 0u; x < RT_SKY_SAMPLING_SIZE; x++) {
		sum += sky_sampling[RT_SKY_SAMPLING_FUNC + row + x];
	}
	row_sums[y] = sum;
	barrier();
	if (y == 0u) {
		float total = 0.0;
		for (uint i = 0u; i < RT_SKY_SAMPLING_SIZE; i++) {
			total += row_sums[i];
		}
		average = total / float(RT_SKY_SAMPLING_SIZE * RT_SKY_SAMPLING_SIZE);
		above_average = false;
	}
	barrier();
	for (uint x = 0u; x < RT_SKY_SAMPLING_SIZE; x++) {
		if (sky_sampling[RT_SKY_SAMPLING_FUNC + row + x] > average) {
			above_average = true; // Benign race: every writer writes true.
		}
	}
	barrier();
	sum = 0.0;
	for (uint x = 0u; x < RT_SKY_SAMPLING_SIZE; x++) {
		float func = sky_sampling[RT_SKY_SAMPLING_FUNC + row + x];
		if (above_average) {
			func = max(func - average, 0.0);
			sky_sampling[RT_SKY_SAMPLING_FUNC + row + x] = func;
		}
		sum += func;
		sky_sampling[RT_SKY_SAMPLING_ROW_CDF + row + x] = sum;
	}
	barrier();
	row_sums[y] = sum;
	barrier();
	if (y == 0u) {
		float total = 0.0;
		for (uint i = 0u; i < RT_SKY_SAMPLING_SIZE; i++) {
			total += row_sums[i];
			sky_sampling[RT_SKY_SAMPLING_MARGINAL_CDF + i] = total;
		}
	}
#else
	const uvec2 texel = gl_GlobalInvocationID.xy;
	if (any(greaterThanEqual(texel, uvec2(RT_SKY_SAMPLING_SIZE)))) {
		return;
	}
	// The texel's average over 4x4 points: the radiance map is finer than the distribution (a sun smaller than a texel
	// still weighs its share). Only a function of the sky: next event estimation is unbiased with any distribution that the
	// BRDF sampling it is combined with covers.
	const uint SUBSAMPLES = 4u;
	float luminance = 0.0;
	for (uint j = 0u; j < SUBSAMPLES; j++) {
		for (uint i = 0u; i < SUBSAMPLES; i++) {
			const vec2 square = (vec2(texel) + (vec2(i, j) + 0.5) / float(SUBSAMPLES)) / float(RT_SKY_SAMPLING_SIZE);
			const vec2 uv = vec3_to_oct_with_border(rt_equal_area_square_to_sphere(square), params.border_size);
#ifdef USE_RADIANCE_OCTMAP_ARRAY
			const vec3 radiance = textureLod(sampler2DArray(radiance_octmap, radiance_sampler), vec3(uv, 0.0), 0.0).rgb;
#else
			const vec3 radiance = textureLod(sampler2D(radiance_octmap, radiance_sampler), uv, 0.0).rgb;
#endif
			luminance += dot(radiance, vec3(0.2126, 0.7152, 0.0722));
		}
	}
	luminance /= float(SUBSAMPLES * SUBSAMPLES);
	// A sky shader's NaN or negative values would break the distribution, not just one direction.
	if (!(luminance > 0.0) || isinf(luminance)) {
		luminance = 0.0;
	}
	sky_sampling[RT_SKY_SAMPLING_FUNC + texel.y * RT_SKY_SAMPLING_SIZE + texel.x] = luminance;
#endif
}

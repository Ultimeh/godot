// The sky's sampling distribution: a piecewise constant 2D distribution of the radiance octmap's luminance (PBRT v4,
// "Image Infinite Lights" and PiecewiseConstant2D), over the equal-area octahedral mapping of the sphere, so every texel
// covers the same solid angle, 4 pi / RT_SKY_SAMPLING_SIZE^2. Built by raytracing_sky_sampling.glsl, sampled by
// raytracing_environment_inc.glsl.
//
// The buffer holds floats, rows along Y:
//   [RT_SKY_SAMPLING_FUNC + y * SIZE + x]: the texel's luminance;
//   [RT_SKY_SAMPLING_ROW_CDF + y * SIZE + x]: the row's luminance summed up to x included;
//   [RT_SKY_SAMPLING_MARGINAL_CDF + y]: the rows' sums summed up to y included (the last one is the total).

#define RT_SKY_SAMPLING_SIZE 128u
#define RT_SKY_SAMPLING_FUNC 0u
#define RT_SKY_SAMPLING_ROW_CDF (RT_SKY_SAMPLING_SIZE * RT_SKY_SAMPLING_SIZE)
#define RT_SKY_SAMPLING_MARGINAL_CDF (2u * RT_SKY_SAMPLING_SIZE * RT_SKY_SAMPLING_SIZE)
#define RT_SKY_SAMPLING_FLOATS (2u * RT_SKY_SAMPLING_SIZE * RT_SKY_SAMPLING_SIZE + RT_SKY_SAMPLING_SIZE)

#ifndef PI
#define PI 3.141592653589f
#endif

// Clarberg, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD" (2008), as PBRT v4's EqualAreaSquareToSphere().
vec3 rt_equal_area_square_to_sphere(vec2 p_square) {
	const float u = 2.0 * p_square.x - 1.0;
	const float v = 2.0 * p_square.y - 1.0;
	const float up = abs(u);
	const float vp = abs(v);
	const float signed_distance = 1.0 - (up + vp);
	const float r = 1.0 - abs(signed_distance);
	const float phi = (r == 0.0 ? 1.0 : (vp - up) / r + 1.0) * (PI / 4.0);
	const float z = (signed_distance >= 0.0 ? 1.0 : -1.0) * (1.0 - r * r);
	const float cos_phi = (u >= 0.0 ? 1.0 : -1.0) * abs(cos(phi));
	const float sin_phi = (v >= 0.0 ? 1.0 : -1.0) * abs(sin(phi));
	const float xy = r * sqrt(max(0.0, 2.0 - r * r));
	return vec3(cos_phi * xy, sin_phi * xy, z);
}

// Its inverse, as PBRT v4's EqualAreaSphereToSquare() (with atan() itself instead of its polynomial).
vec2 rt_equal_area_sphere_to_square(vec3 p_dir) {
	const vec3 a = abs(p_dir);
	const float r = sqrt(max(0.0, 1.0 - a.z));
	const float m = max(a.x, a.y);
	const float b = m == 0.0 ? 0.0 : min(a.x, a.y) / m;
	float phi = atan(b) * (2.0 / PI);
	if (a.x < a.y) {
		phi = 1.0 - phi;
	}
	float v = phi * r;
	float u = r - v;
	if (p_dir.z < 0.0) {
		const float t = u;
		u = 1.0 - v;
		v = 1.0 - t;
	}
	u = p_dir.x >= 0.0 ? u : -u;
	v = p_dir.y >= 0.0 ? v : -v;
	return vec2(u, v) * 0.5 + 0.5;
}

uint rt_sky_sampling_texel(vec2 p_square) {
	const uvec2 texel = min(uvec2(p_square * float(RT_SKY_SAMPLING_SIZE)), uvec2(RT_SKY_SAMPLING_SIZE - 1u));
	return texel.y * RT_SKY_SAMPLING_SIZE + texel.x;
}

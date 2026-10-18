// Decals at a closest hit, applied over the material as raster's scene_forward_clustered.glsl applies them, in world space.
// Requires: raytracing_closest_hit_common_inc.glsl (HitData, geometries[], rt_params), raytracing_material_eval_inc.glsl.

// Matches C++ RT_DecalData (render_raytracing.h, 224 bytes).
struct RTDecalData {
	vec4 xform[3]; // World to decal space, rows: x and z in [0, 1], y in [-1, 1] inside its box.
	vec4 albedo_rect; // In the decal atlas; zero: no texture.
	vec4 normal_rect;
	vec4 orm_rect;
	vec4 emission_rect;
	vec4 modulate; // Linear, its alpha times the distance fade.
	float emission_energy; // Times the distance fade.
	uint mask; // Decal.cull_mask, against the instance's layers.
	float upper_fade;
	float lower_fade;
	vec4 normal_xform[3]; // The decal's orthonormalized basis, rows: its normal map to world space.
	vec3 normal; // Its Y axis, world space.
	float normal_fade;
	float albedo_mix;
	uint pad0;
	uint pad1;
	uint pad2;
};

// clang-format off
layout(set = 0, binding = 33, std430) readonly buffer RTDecalBuffer {
	RTDecalData rt_decals[];
};

// RenderRaytracing::gather_decals(): [0] 1 / cell size (float bits), [1] buckets - 1, [2] G, the decals tested at every hit,
// [3] unused; [4, 4 + G) those decals; [4 + G, 4 + G + buckets] the first entry of each bucket in this buffer (the last
// one ends the previous), then the entries. Indices ascend in every list: farthest decal first, the nearest on top.
layout(set = 0, binding = 34, std430) readonly buffer RTDecalGrid {
	uint rt_decal_grid[];
};
// clang-format on

layout(set = 0, binding = 35) uniform texture2D rt_decal_atlas;
layout(set = 0, binding = 36) uniform texture2D rt_decal_atlas_srgb;
layout(set = 0, binding = 37) uniform sampler rt_decal_sampler;

// Same as _rt_decal_grid_hash() in render_raytracing.cpp.
uint rt_decal_grid_hash(ivec3 p_cell) {
	return (uint(p_cell.x) * 73856093u) ^ (uint(p_cell.y) * 19349663u) ^ (uint(p_cell.z) * 83492791u);
}

/// The world position's screen derivatives at the hit, what raster's dFdx(vertex) / dFdy(vertex) see.
void rt_decal_position_derivatives(HitData h, out vec3 r_ddx, out vec3 r_ddy) {
	r_ddx = vec3(0.0);
	r_ddy = vec3(0.0);
	GeometryData geom = geometries[h.geometry_idx];
	if ((geom.flags & FLAG_PROCEDURAL) != 0u) {
		return;
	}
	uint i0, i1, i2;
	get_triangle_indices(geom, i0, i1, i2);
	vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
	vec3 bary_ddx, bary_ddy;
	hit_barycentric_derivatives(geom, i0, i1, i2, bary, bary_ddx, bary_ddy);
	vec3 p0 = gl_ObjectToWorldEXT * vec4(fetch_blas_position(geom, i0), 1.0);
	vec3 e1 = gl_ObjectToWorldEXT * vec4(fetch_blas_position(geom, i1), 1.0) - p0;
	vec3 e2 = gl_ObjectToWorldEXT * vec4(fetch_blas_position(geom, i2), 1.0) - p0;
	// The barycentrics' derivatives sum to zero: from the edges, without the position's magnitude.
	r_ddx = e1 * bary_ddx.y + e2 * bary_ddx.z;
	r_ddy = e1 * bary_ddy.y + e2 * bary_ddy.z;
}

vec4 rt_decal_sample(vec4 p_rect, vec3 p_uv_local, vec2 p_ddx, vec2 p_ddy) {
	return textureGrad(sampler2D(rt_decal_atlas, rt_decal_sampler), p_uv_local.xz * p_rect.zw + p_rect.xy, p_ddx * p_rect.zw, p_ddy * p_rect.zw);
}

vec4 rt_decal_sample_srgb(vec4 p_rect, vec3 p_uv_local, vec2 p_ddx, vec2 p_ddy) {
	return textureGrad(sampler2D(rt_decal_atlas_srgb, rt_decal_sampler), p_uv_local.xz * p_rect.zw + p_rect.xy, p_ddx * p_rect.zw, p_ddy * p_rect.zw);
}

/// The decals over this hit, on its material. p_geo_normal: the normal before the normal map (raster's geo_normal), world
/// space. A sampler without mipmaps has no level past 0, so textureGrad() covers every decal filter.
void rt_apply_decals(HitData h, vec3 p_geo_normal, inout MaterialResult m) {
	if (get_rt_param(RT_PARAM_DECAL_COUNT) <= 0.0) {
		return;
	}
	vec3 pos = h.hit_pos;
	ivec3 cell = ivec3(floor(pos * uintBitsToFloat(rt_decal_grid[0])));
	uint global_end = 4u + rt_decal_grid[2];
	uint bucket = global_end + (rt_decal_grid_hash(cell) & rt_decal_grid[1]);
	uint gi = 4u;
	uint bi = rt_decal_grid[bucket];
	uint bucket_end = rt_decal_grid[bucket + 1u];
	if (gi == global_end && bi == bucket_end) {
		return;
	}

	uint layer_mask = geometries[h.geometry_idx].layer_mask;
	bool have_derivatives = false;
	vec3 pos_ddx = vec3(0.0);
	vec3 pos_ddy = vec3(0.0);

	// The two ascending lists merged, in raster's order.
	while (gi < global_end || bi < bucket_end) {
		uint g = gi < global_end ? rt_decal_grid[gi] : 0xFFFFFFFFu;
		uint b = bi < bucket_end ? rt_decal_grid[bi] : 0xFFFFFFFFu;
		uint idx;
		if (g < b) {
			idx = g;
			gi++;
		} else {
			idx = b;
			bi++;
		}

		if ((rt_decals[idx].mask & layer_mask) == 0u) {
			continue;
		}

		vec4 pos4 = vec4(pos, 1.0);
		vec3 uv_local = vec3(dot(rt_decals[idx].xform[0], pos4), dot(rt_decals[idx].xform[1], pos4), dot(rt_decals[idx].xform[2], pos4));
		if (any(lessThan(uv_local, vec3(0.0, -1.0, 0.0))) || any(greaterThan(uv_local, vec3(1.0)))) {
			continue;
		}

		float fade = pow(1.0 - (uv_local.y > 0.0 ? uv_local.y : -uv_local.y), uv_local.y > 0.0 ? rt_decals[idx].upper_fade : rt_decals[idx].lower_fade);
		if (rt_decals[idx].normal_fade > 0.0) {
			fade *= smoothstep(rt_decals[idx].normal_fade, 1.0, dot(p_geo_normal, rt_decals[idx].normal) * 0.5 + 0.5);
		}

		if (!have_derivatives) {
			rt_decal_position_derivatives(h, pos_ddx, pos_ddy);
			have_derivatives = true;
		}
		vec2 ddx = vec2(dot(rt_decals[idx].xform[0].xyz, pos_ddx), dot(rt_decals[idx].xform[2].xyz, pos_ddx));
		vec2 ddy = vec2(dot(rt_decals[idx].xform[0].xyz, pos_ddy), dot(rt_decals[idx].xform[2].xyz, pos_ddy));

		if (rt_decals[idx].albedo_rect != vec4(0.0)) {
			vec4 decal_albedo = rt_decal_sample_srgb(rt_decals[idx].albedo_rect, uv_local, ddx, ddy);
			decal_albedo *= rt_decals[idx].modulate;
			decal_albedo.a *= fade;
			m.albedo = mix(m.albedo, decal_albedo.rgb, decal_albedo.a * rt_decals[idx].albedo_mix);

			if (rt_decals[idx].normal_rect != vec4(0.0)) {
				vec3 decal_normal = rt_decal_sample(rt_decals[idx].normal_rect, uv_local, ddx, ddy).xyz;
				decal_normal.xy = decal_normal.xy * vec2(2.0, -2.0) - vec2(1.0, -1.0); // Flipped Y, as raster.
				decal_normal.z = sqrt(max(0.0, 1.0 - dot(decal_normal.xy, decal_normal.xy)));
				vec3 n = decal_normal.xzy; // Y is up.
				decal_normal = vec3(dot(rt_decals[idx].normal_xform[0].xyz, n), dot(rt_decals[idx].normal_xform[1].xyz, n), dot(rt_decals[idx].normal_xform[2].xyz, n));
				m.normal = normalize(mix(m.normal, decal_normal, decal_albedo.a));
			}

			if (rt_decals[idx].orm_rect != vec4(0.0)) {
				vec3 decal_orm = rt_decal_sample(rt_decals[idx].orm_rect, uv_local, ddx, ddy).xyz;
				m.ao = mix(m.ao, decal_orm.r, decal_albedo.a);
				m.roughness = mix(m.roughness, decal_orm.g, decal_albedo.a);
				m.metalness = mix(m.metalness, decal_orm.b, decal_albedo.a);
			}
		}

		if (rt_decals[idx].emission_rect != vec4(0.0)) {
			// Additive, independent of the albedo; MaterialResult's emission is exposure-normalized already.
			m.emissive += rt_decal_sample_srgb(rt_decals[idx].emission_rect, uv_local, ddx, ddy).rgb * rt_decals[idx].modulate.rgb * rt_decals[idx].emission_energy * fade * scene_data_block.data.emissive_exposure_normalization;
		}
	}
}

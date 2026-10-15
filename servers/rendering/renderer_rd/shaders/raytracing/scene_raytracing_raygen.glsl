#[raygen]

#version 460

#extension GL_EXT_control_flow_attributes : enable

#VERSION_DEFINES

// clang-format off
#include "raytracing_inc.glsl"
#include "../scene_data_inc.glsl"
// clang-format on

#pragma shader_stage(raygen)
#extension GL_EXT_ray_tracing : enable
#ifdef USE_SER
#extension GL_EXT_shader_invocation_reorder : enable
#endif

#define GLSL 1
#define RT_STAGE_RAYGEN 1
#include "raytracing_common_inc.glsl"

// The path tracer's output, RB_TEX_RAYTRACING: its format, or the writes are undefined on Vulkan (D3D12 converts them).
layout(set = 0, binding = 0, rgba16f) uniform image2D image;
// The reflection of the first transparent surface in front of the opaque ones, and that surface's view depth (0: none).
layout(set = 0, binding = 29, rgba16f) uniform image2D front_layer_image;
layout(set = 0, binding = 1) uniform accelerationStructureEXT tlas;

layout(location = 0) rayPayloadEXT PathPayload payload;

void main() {
	uvec2 pixel = gl_LaunchIDEXT.xy;
	const vec2 pixel_center = vec2(pixel) + vec2(0.5);
	const vec2 in_uv = pixel_center / vec2(gl_LaunchSizeEXT.xy);
	vec2 d = in_uv * 2.0 - 1.0;

	mat4 inv_view = transpose(mat4(scene_data_block.data.inv_view_matrix[0],
			scene_data_block.data.inv_view_matrix[1],
			scene_data_block.data.inv_view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));

	vec4 target = scene_data_block.data.inv_projection_matrix * vec4(d.x, d.y, 1.0, 1.0);
	vec4 origin = inv_view * vec4(0.0, 0.0, 0.0, 1.0);
	vec4 direction = inv_view * vec4(normalize(target.xyz), 0);

	// Sample count from specialization constant, frame index from uniform
	const uint samples_per_pixel = RT_GET_SAMPLE_COUNT();
	uint frame_index = uint(get_rt_param(RT_PARAM_FRAME_INDEX));

	// Accumulate multiple samples per pixel
	vec3 total_radiance = vec3(0.0);
	float opaque_t = 10000.0;

	const uint max_bounces = RT_GET_MAX_BOUNCES();

	// TODO: when we have a spp > 0 the first raycast is always identical,
	// we should move it out of the loop

	[[dont_unroll]] for (uint sample_idx = 0u; sample_idx < samples_per_pixel; sample_idx++) {
		PathState ps;
		ps.radiance = vec3(0.0);
		ps.throughput = vec3(1.0);
		ps.packed_bounces_flags = (sample_idx == 0u) ? set_sample_zero(0u) : 0u;
		ps.rng_state = init_rng(pixel, frame_index, sample_idx);

		vec3 ray_origin = origin.xyz;
		vec3 ray_dir = direction.xyz;

		[[dont_unroll]] for (uint bounce = 0u; bounce <= max_bounces; bounce++) {
			path_pack(payload, ps);

#ifdef USE_SER
			hitObjectEXT hitObject;
			hitObjectTraceRayEXT(hitObject, tlas, RT_RAY_FLAGS, RT_MASK_CAMERA, 0, 0, 0, ray_origin, 0.001, ray_dir, 10000.0, 0);

			// Reorder with a coherence hint that has 8 bits
			uint hint = 0;
			if (hitObjectIsHitEXT(hitObject)) {
				// TODO: This hint barely does anything. There is a lot of untapped potential here.
				hint = hitObjectGetInstanceIdEXT(hitObject);
			}
			reorderThreadEXT(hitObject, hint, 8);

			hitObjectExecuteShaderEXT(hitObject, 0);
#else
			traceRayEXT(tlas, RT_RAY_FLAGS, RT_MASK_CAMERA, 0, 0, 0, ray_origin, 0.001, ray_dir, 10000.0, 0);
#endif

			ps = path_unpack(payload);
			if (sample_idx == 0u && bounce == 0u && !is_path_terminated(ps.packed_bounces_flags)) {
				opaque_t = ps.hit_t;
			}
			if (is_path_terminated(ps.packed_bounces_flags)) {
				break;
			}
			// Reconstruct the next ray origin from the current ray + hit distance, apply bias
			vec3 hit_pos = ray_origin + ray_dir * ps.hit_t;
			ray_origin = offset_ray_origin(hit_pos, ps.offset_normal);
			ray_dir = ps.next_ray_dir;
		}

		total_radiance += ps.radiance;
	}

	vec3 final_radiance = total_radiance / float(samples_per_pixel);

	imageStore(image, ivec2(pixel), vec4(final_radiance, 1.0));

	// Front layer: the first transparent surface before the opaque hit, and one path along its reflection, for the raster
	// transparent pass (RenderForwardClustered::front_layer_reflection). Not a primary ray: no depth, motion or RR guides.
	vec4 front_layer = vec4(0.0);
	if (get_rt_param(RT_PARAM_FRONT_LAYER_COUNT) > 0.0) {
		PathState fs;
		fs.radiance = vec3(0.0);
		fs.throughput = vec3(1.0);
		fs.packed_bounces_flags = set_front_layer_ray(0u);
		fs.rng_state = init_rng(pixel, frame_index, samples_per_pixel);
		fs.hit_t = opaque_t; // Read by the hit shaders as DEPTH_TEXTURE (rt_depth_buffer_value()).
		fs.offset_normal = vec3(0.0, 0.0, 1.0);
		fs.next_ray_dir = vec3(0.0, 0.0, 1.0);
		path_pack(payload, fs);
		traceRayEXT(tlas, RT_RAY_FLAGS, RT_MASK_FRONT_LAYER, 0, 0, 0, origin.xyz, 0.001, direction.xyz, max(opaque_t, 0.001), 0);
		fs = path_unpack(payload);

		if (!is_path_terminated(fs.packed_bounces_flags)) {
			float view_depth = fs.hit_t * dot(direction.xyz, -normalize(inv_view[2].xyz));

			PathState rs;
			rs.radiance = vec3(0.0);
			rs.throughput = vec3(1.0);
			rs.packed_bounces_flags = pack_bounces(0u, 0u); // As deep as a camera path; not sample zero: no primary writes.
			rs.rng_state = fs.rng_state;
			vec3 ray_origin = offset_ray_origin(origin.xyz + direction.xyz * fs.hit_t, fs.offset_normal);
			vec3 ray_dir = fs.next_ray_dir;
			// The reflected point's direct light and one indirect bounce, as Lumen's front layer reflections: a full path
			// for every pixel of a window close to the camera doubled the cost of the trace.
			uint front_layer_bounces = min(max_bounces, 1u);
			[[dont_unroll]] for (uint bounce = 0u; bounce <= front_layer_bounces; bounce++) {
				path_pack(payload, rs);
#ifdef USE_SER
				// Reflections off a window diverge as much as camera bounces: regrouped by what they hit, as above.
				hitObjectEXT hitObject;
				hitObjectTraceRayEXT(hitObject, tlas, RT_RAY_FLAGS, RT_MASK_CAMERA, 0, 0, 0, ray_origin, 0.001, ray_dir, 10000.0, 0);
				reorderThreadEXT(hitObject, hitObjectIsHitEXT(hitObject) ? hitObjectGetInstanceIdEXT(hitObject) : 0u, 8);
				hitObjectExecuteShaderEXT(hitObject, 0);
#else
				traceRayEXT(tlas, RT_RAY_FLAGS, RT_MASK_CAMERA, 0, 0, 0, ray_origin, 0.001, ray_dir, 10000.0, 0);
#endif
				rs = path_unpack(payload);
				if (is_path_terminated(rs.packed_bounces_flags)) {
					break;
				}
				vec3 hit_pos = ray_origin + ray_dir * rs.hit_t;
				ray_origin = offset_ray_origin(hit_pos, rs.offset_normal);
				ray_dir = rs.next_ray_dir;
			}
			front_layer = vec4(rs.radiance, view_depth);
		}
	}
	imageStore(front_layer_image, ivec2(pixel), front_layer);
}

#[miss]

#version 460

#VERSION_DEFINES

#pragma shader_stage(miss)
#extension GL_EXT_ray_tracing : enable

#define GLSL 1
#define RT_STAGE_MISS 1

// clang-format off
#include "raytracing_inc.glsl"
#include "../scene_data_inc.glsl"
#include "brdf_inc.glsl"
#include "raytracing_common_inc.glsl"
// clang-format on

layout(location = 0) rayPayloadInEXT PathPayload payload;

#ifdef USE_RADIANCE_OCTMAP_ARRAY

layout(set = 0, binding = 7) uniform texture2DArray radiance_octmap;
layout(set = 0, binding = 8) uniform sampler radiance_sampler;

// Array-backed octahedral radiance: roughness selects the prefiltered array
// layer (with a blend between adjacent layers). Ray/hit stages have no
// screen-space derivatives, so the intra-layer mip is fixed at 0.0; roughness is
// represented entirely by the layer selection.
vec3 radiance_octmap_sample(vec2 p_oct_uv, float p_roughness) {
	float layer;
	float blend = modf(clamp(p_roughness, 0.0, 1.0) * MAX_ROUGHNESS_LOD, layer);
	vec3 a = textureLod(sampler2DArray(radiance_octmap, radiance_sampler), vec3(p_oct_uv, layer), 0.0).rgb;
	vec3 b = textureLod(sampler2DArray(radiance_octmap, radiance_sampler), vec3(p_oct_uv, layer + 1.0), 0.0).rgb;
	return mix(a, b, blend);
}

#else

layout(set = 0, binding = 7) uniform texture2D radiance_octmap;
layout(set = 0, binding = 8) uniform sampler radiance_sampler;

// Single-texture octahedral radiance: roughness maps to the mip LOD.
vec3 radiance_octmap_sample(vec2 p_oct_uv, float p_roughness) {
	return textureLod(sampler2D(radiance_octmap, radiance_sampler), p_oct_uv, clamp(p_roughness, 0.0, 1.0) * MAX_ROUGHNESS_LOD).rgb;
}

#endif // USE_RADIANCE_OCTMAP_ARRAY

void main() {
	PathState ps = path_unpack(payload);

#if !defined(USE_SER)
	// Shadow rays that miss mean the light is visible (no occluder).
	if (is_shadow_ray(ps.packed_bounces_flags)) {
		ps.radiance = vec3(1.0);
		path_pack(payload, ps);
		return;
	}
#endif

	// Miss always ends the path.
	ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);

#ifdef RT_DEBUG_ENABLED
	{
		int VIS_MODE = int(get_rt_param(RT_PARAM_VIS_MODE));
		// Specular hit distance: the first-bounce closest_hit pre-seeds the
		// "no hit" color into radiance, so a missed reflection just keeps it.
		if (VIS_MODE == 13 && get_total_bounces(ps.packed_bounces_flags) > 0u) {
			path_pack(payload, ps);
			return;
		}
	}
#endif // RT_DEBUG_ENABLED

	// Primary ray miss: write depth, velocity, and DLSS RR defaults (sample 0 only).
	{
		uint total_bounces = get_total_bounces(ps.packed_bounces_flags);
		if (total_bounces == 0u && is_sample_zero(ps.packed_bounces_flags)) {
			ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);

			imageStore(rt_depth_image, pixel, vec4(0.0));

			// Sky velocity: reproject a far-plane point using unjittered VPs.
			{
				vec3 far_world = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * 10000.0;
				vec2 curr_uv = project_uv(far_world, curr_vp_unjittered);
				vec2 prev_uv = project_uv(far_world, prev_vp_unjittered);
				imageStore(rt_velocity_image, pixel, vec4(prev_uv - curr_uv, 0.0, 0.0));
			}
		}
	}

	mat3 camera_basis = mat3(scene_data_block.data.inv_view_matrix);
	mat3 world_to_sky = scene_data_block.data.radiance_inverse_xform * camera_basis;
	vec3 sky_dir = world_to_sky * gl_WorldRayDirectionEXT;

	vec2 border = vec2(scene_data_block.data.radiance_border_size,
			1.0 - scene_data_block.data.radiance_border_size * 2.0);
	vec2 sky_uv = vec3_to_oct_with_border(sky_dir, border);

	vec3 sky_color = radiance_octmap_sample(sky_uv, 0.0);
	sky_color *= scene_data_block.data.IBL_exposure_normalization;

	if ((RT_FLAGS & RT_FLAG_FOG_ENABLED) != 0u) {
		vec3 fog_color = scene_data_block.data.fog_light_color;

		if (scene_data_block.data.fog_aerial_perspective > 0.0) {
			vec3 sky_fog = radiance_octmap_sample(sky_uv, 1.0 / MAX_ROUGHNESS_LOD);
			sky_fog *= scene_data_block.data.IBL_exposure_normalization;
			fog_color = mix(fog_color, sky_fog, scene_data_block.data.fog_aerial_perspective);
		}

		sky_color = mix(sky_color, fog_color, scene_data_block.data.fog_sky_affect);
	}

#ifdef DLSS_RR_ENABLED
	{
		uint total_bounces = get_total_bounces(ps.packed_bounces_flags);
		if (total_bounces == 0u && is_sample_zero(ps.packed_bounces_flags)) {
			ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);
			imageStore(dlss_rr_diffuse_albedo, pixel, vec4(DLSSRR_encodeDiffuseAlbedo(sky_color), 1.0));
			imageStore(dlss_rr_specular_albedo, pixel, vec4(0.0));
			imageStore(dlss_rr_normal_roughness, pixel, vec4(-gl_WorldRayDirectionEXT, 0.0));
			imageStore(dlss_rr_specular_hit_dist, pixel, vec4(-1.0));
		}
	}
#endif

#ifdef RT_DEBUG_ENABLED
	{
		int VIS_MODE = int(get_rt_param(RT_PARAM_VIS_MODE));
		if (VIS_MODE == 20) {
			ps.radiance = vec3(1.0);
		} else if (VIS_MODE == 0) {
			ps.radiance += ps.throughput * sky_color;
		} else {
			ps.radiance = sky_color;
		}
	}
#else
	ps.radiance += ps.throughput * sky_color;
#endif // RT_DEBUG_ENABLED

	path_pack(payload, ps);
}

#[closest_hit]

#version 460

#VERSION_DEFINES

#pragma shader_stage(closest_hit)
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_query : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_nonuniform_qualifier : require
#ifdef USE_SER
#extension GL_EXT_shader_invocation_reorder : enable
#endif

#define GLSL 1
#define RT_STAGE_CLOSEST_HIT 1

// clang-format off
#include "raytracing_inc.glsl"
#include "../scene_data_inc.glsl"
#include "brdf_inc.glsl"
#include "raytracing_common_inc.glsl"
// clang-format on

#define attribs hit_attribs.bary_or_uv
#define RT_HIT_ATTRIBS_DECLARED

#include "raytracing_hit_inc.glsl"

layout(set = 0, binding = 1) uniform accelerationStructureEXT tlas;
layout(location = 0) rayPayloadInEXT PathPayload payload;

layout(set = 1, binding = 0) uniform texture2D bindless_textures[];
layout(set = 2, binding = 0) uniform texture2DArray bindless_texture_arrays[];

#include "raytracing_samplers_inc.glsl"

// clang-format off
layout(set = 0, binding = 3, std430) readonly buffer GeometryBuffer {
	GeometryData geometries[];
};

layout(set = 0, binding = 4, std430) readonly buffer MotionIndexBuffer {
	int motion_indices[];
};

layout(set = 0, binding = 5, std430) readonly buffer MaterialBuffer {
	MaterialData materials[];
};
// clang-format on

#include "raytracing_lights_inc.glsl"

// clang-format off
layout(set = 0, binding = 32, std430) readonly buffer MotionTransforms {
	InstanceMotionData motion_transforms[];
};
// clang-format on

#ifdef USE_RADIANCE_OCTMAP_ARRAY

layout(set = 0, binding = 7) uniform texture2DArray radiance_octmap;
layout(set = 0, binding = 8) uniform sampler radiance_sampler;

// Array-backed octahedral radiance: roughness selects the prefiltered array
// layer (with a blend between adjacent layers). Ray/hit stages have no
// screen-space derivatives, so the intra-layer mip is fixed at 0.0; roughness is
// represented entirely by the layer selection.
vec3 radiance_octmap_sample(vec2 p_oct_uv, float p_roughness) {
	float layer;
	float blend = modf(clamp(p_roughness, 0.0, 1.0) * MAX_ROUGHNESS_LOD, layer);
	vec3 a = textureLod(sampler2DArray(radiance_octmap, radiance_sampler), vec3(p_oct_uv, layer), 0.0).rgb;
	vec3 b = textureLod(sampler2DArray(radiance_octmap, radiance_sampler), vec3(p_oct_uv, layer + 1.0), 0.0).rgb;
	return mix(a, b, blend);
}

#else

layout(set = 0, binding = 7) uniform texture2D radiance_octmap;
layout(set = 0, binding = 8) uniform sampler radiance_sampler;

// Single-texture octahedral radiance: roughness maps to the mip LOD.
vec3 radiance_octmap_sample(vec2 p_oct_uv, float p_roughness) {
	return textureLod(sampler2D(radiance_octmap, radiance_sampler), p_oct_uv, clamp(p_roughness, 0.0, 1.0) * MAX_ROUGHNESS_LOD).rgb;
}

#endif // USE_RADIANCE_OCTMAP_ARRAY

// clang-format off
#include "raytracing_material_eval_inc.glsl"
#include "raytracing_closest_hit_common_inc.glsl"
// clang-format on

// ============================================================================
// CUSTOM SHADER GLOBALS (injected by ShaderCompiler for HG1+)
// ============================================================================
#ifdef RT_CUSTOM_HIT_GROUP
#include "raytracing_custom_globals_inc.glsl"
#endif

// ============================================================================
// MAIN
// ============================================================================
void main() {
	HitData h = compute_hit_data();
	// Every variant indexes the texture array table dynamically, so the descriptor stays unbounded in each layout.
	if (h.geometry_idx == 0xFFFFFFFFu) {
		h.hit_pos.x += textureLod(sampler2DArray(bindless_texture_arrays[nonuniformEXT(h.geometry_idx)], SAMPLER_LINEAR_REPEAT), vec3(0.0), 0.0).r;
	}
	write_primary_hit_depth(h.hit_pos);
	write_primary_hit_velocity(h.hit_pos);

#ifdef RT_CUSTOM_HIT_GROUP
	uint rt_geometry_idx = h.geometry_idx;
	vec3 rt_hit_pos = h.hit_pos;
	uint rt_i0, rt_i1, rt_i2;
	get_triangle_indices(geometries[h.geometry_idx], rt_i0, rt_i1, rt_i2);
	vec3 rt_bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
	vec2 rt_uv = h.uv;
	vec4 rt_color = h.color;
	vec3 rt_normal = h.geometry_normal;
	vec3 rt_tangent = h.tangent;
	vec3 rt_bitangent = h.bitangent;
	bool rt_front_face = h.is_front_face;

	// A point vertex() removed (NaN position) lets the ray pass through, as raster draws nothing there.
#define RT_CUSTOM_REMOVED_POINT                             \
	{                                                       \
		PathState removed_ps = path_unpack(payload);        \
		removed_ps.hit_t = gl_HitTEXT;                      \
		removed_ps.offset_normal = gl_WorldRayDirectionEXT; \
		removed_ps.next_ray_dir = gl_WorldRayDirectionEXT;  \
		path_pack(payload, removed_ps);                     \
		return;                                             \
	}
#include "raytracing_custom_fragment_inc.glsl"

	// Build MaterialResult from fragment outputs.
	MaterialResult m;
	m.albedo = albedo;
	m.alpha = alpha;
	m.roughness = roughness;
	m.metalness = metallic;
	m.specular = specular;
	m.emissive = emission * scene_data_block.data.emissive_exposure_normalization;
#ifdef MODE_UNSHADED
	m.unshaded = true;
#else
	m.unshaded = false;
#endif
	m.ao = clamp(ao, 0.0, 1.0);
	m.ao_light_affect = clamp(ao_light_affect, 0.0, 1.0);
	m.backlight = max(backlight, vec3(0.0));
#ifdef SPECULAR_DISABLED
	m.specular_disabled = true;
#else
	m.specular_disabled = false;
#endif
	m.normal = normalize(mat3(inv_view_matrix) * normal); // view space -> world space
	if (any(isnan(m.normal))) {
		m.normal = h.geometry_normal; // Material code can leave NaN normals (screen derivatives are zero here).
	}

	// NORMAL_MAP as raster applies it: in the fragment's own TANGENT, BINORMAL and NORMAL, which a material can compute
	// itself (Terrain3D derives them from its heightmap; the mesh's tangent frame would turn its relief inside out).
	if (normal_map != vec3(0.5, 0.5, 1.0)) {
		vec3 ts_normal;
		ts_normal.xy = normal_map.xy * 2.0 - 1.0;
		ts_normal.z = sqrt(max(0.0, 1.0 - dot(ts_normal.xy, ts_normal.xy)));
		vec3 mapped = normalize(mix(normal, tangent * ts_normal.x + binormal * ts_normal.y + normal * ts_normal.z, normal_map_depth));
		m.normal = normalize(mat3(inv_view_matrix) * mapped);
		if (any(isnan(m.normal))) {
			m.normal = h.geometry_normal;
		}
	}

#ifdef RT_DEBUG_ENABLED
	{
		int VIS_MODE = int(get_rt_param(RT_PARAM_VIS_MODE));
		if (VIS_MODE != 0) {
			vec3 V = -gl_WorldRayDirectionEXT;
			float NdotV = max(dot(m.normal, V), 0.0001);
			vec3 orm = vec3(1.0, m.roughness, m.metalness);
			debug_visualize(VIS_MODE, h.geometry_normal, m.normal, normal_map,
					h.tangent, h.bitangent, h.uv, m.albedo, orm, m.metalness, m.roughness, m.specular, m.emissive, V, NdotV);
			return;
		}
	}
#endif // RT_DEBUG_ENABLED
	shade_and_bounce(h, m);
#else
	// HG0: StandardMaterial3D evaluation.
	MaterialData mat = materials[h.geometry_idx];
	vec2 uv = h.uv * mat.uv1_scale + mat.uv1_offset;

	// Screen derivatives of the UVs, for the mip level raster picks.
	GeometryData geom = geometries[h.geometry_idx];
	uint i0, i1, i2;
	get_triangle_indices(geom, i0, i1, i2);
	vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
	vec3 bary_ddx, bary_ddy;
	hit_barycentric_derivatives(geom, i0, i1, i2, bary, bary_ddx, bary_ddy);
	vec2 uv0 = fetch_uv(geom, i0, i0, i0, vec3(1.0, 0.0, 0.0));
	vec2 uv1 = fetch_uv(geom, i1, i1, i1, vec3(1.0, 0.0, 0.0));
	vec2 uv2 = fetch_uv(geom, i2, i2, i2, vec3(1.0, 0.0, 0.0));
	vec2 uv_ddx = (uv0 * bary_ddx.x + uv1 * bary_ddx.y + uv2 * bary_ddx.z) * mat.uv1_scale;
	vec2 uv_ddy = (uv0 * bary_ddy.x + uv1 * bary_ddy.y + uv2 * bary_ddy.z) * mat.uv1_scale;

	// Triplanar mapping, set up as BaseMaterial3D's vertex() does: object space unless world triplanar.
	bool triplanar = (mat.flags & MAT_FLAG_UV1_TRIPLANAR) != 0u;
	vec3 triplanar_weights = vec3(0.0);
	vec3 triplanar_pos = vec3(0.0);
	vec3 triplanar_ddx = vec3(0.0);
	vec3 triplanar_ddy = vec3(0.0);
	HitData nh = h;
	if (triplanar) {
		bool world = (mat.flags & MAT_FLAG_UV1_WORLD_TRIPLANAR) != 0u;
		vec3 n = world ? h.geometry_normal : h.mesh_normal;
		vec3 pos = world ? h.hit_pos : h.mesh_pos;
		triplanar_weights = pow(abs(n), vec3(mat.uv1_blend_sharpness));
		triplanar_weights /= dot(triplanar_weights, vec3(1.0));
		triplanar_pos = pos * vec3(mat.uv1_scale, mat.uv1_scale_z) + vec3(mat.uv1_offset, mat.uv1_offset_z);
		triplanar_pos *= vec3(1.0, -1.0, 1.0);
		vec3 p0 = fetch_mesh_position(geom, i0);
		vec3 p1 = fetch_mesh_position(geom, i1);
		vec3 p2 = fetch_mesh_position(geom, i2);
		if (world) {
			mat4 mesh_to_world = mat4(gl_ObjectToWorldEXT);
			if ((geom.flags & FLAG_COMPRESSED) != 0u) {
				mat4 aabb_xform, inv_aabb_xform;
				get_aabb_compression_xforms(geom, aabb_xform, inv_aabb_xform);
				mesh_to_world = mesh_to_world * inv_aabb_xform;
			}
			p0 = (mesh_to_world * vec4(p0, 1.0)).xyz;
			p1 = (mesh_to_world * vec4(p1, 1.0)).xyz;
			p2 = (mesh_to_world * vec4(p2, 1.0)).xyz;
		}
		vec3 scale = vec3(mat.uv1_scale, mat.uv1_scale_z) * vec3(1.0, -1.0, 1.0);
		triplanar_ddx = (p0 * bary_ddx.x + p1 * bary_ddx.y + p2 * bary_ddx.z) * scale;
		triplanar_ddy = (p0 * bary_ddy.x + p1 * bary_ddy.y + p2 * bary_ddy.z) * scale;

		vec3 an = abs(n);
		vec3 tangent = normalize(vec3(0.0, 0.0, -1.0) * an.x + vec3(1.0, 0.0, 0.0) * (an.y + an.z));
		vec3 binormal = normalize(vec3(0.0, 1.0, 0.0) * (an.x + an.z) + vec3(0.0, 0.0, -1.0) * an.y);
		if (!world) {
			// Same rotation compute_hit_data() applies to the mesh's tangents.
			mat3 model_rotation = mat3(gl_ObjectToWorldEXT);
			mat3 normal_matrix = mat3(normalize(model_rotation[0]), normalize(model_rotation[1]), normalize(model_rotation[2]));
			tangent = normal_matrix * tangent;
			binormal = normal_matrix * binormal;
		}
		nh.tangent = normalize(tangent);
		nh.bitangent = normalize(binormal);
	}

	// Normal mapping.
	vec3 tangent_space_normal = vec3(0.0, 0.0, 1.0);
	vec3 final_normal = h.geometry_normal;
	if ((mat.flags & 1u) != 0u) {
		// With the material's filter, as raster samples every texture of a BaseMaterial3D.
		vec3 normal_sample = triplanar ? sample_material_triplanar(mat.normal_texture_idx, triplanar_weights, triplanar_pos, triplanar_ddx, triplanar_ddy, mat.flags).rgb : sample_material_texture_grad(mat.normal_texture_idx, uv, uv_ddx, uv_ddy, mat.flags).rgb;
		tangent_space_normal.xy = normal_sample.xy * 2.0 - 1.0;
		tangent_space_normal.z = sqrt(max(0.0, 1.0 - dot(tangent_space_normal.xy, tangent_space_normal.xy)));
		final_normal = apply_normal_map(nh, tangent_space_normal, mat.normal_map_depth);
	}

	// Texture sampling.
	vec4 albedo_tex = triplanar ? sample_material_triplanar(mat.albedo_texture_idx, triplanar_weights, triplanar_pos, triplanar_ddx, triplanar_ddy, mat.flags) : sample_material_texture_grad(mat.albedo_texture_idx, uv, uv_ddx, uv_ddy, mat.flags);
	if ((mat.flags & MAT_FLAG_VERTEX_COLOR_ALBEDO) != 0u) {
		albedo_tex *= material_vertex_color(fetch_color(geom, i0, i1, i2, bary), mat.flags); // HitData skips the color here.
	}
	vec3 albedo = albedo_tex.rgb * mat.albedo_color.rgb;
	vec3 orm = (triplanar ? sample_material_triplanar(mat.orm_texture_idx, triplanar_weights, triplanar_pos, triplanar_ddx, triplanar_ddy, mat.flags) : sample_material_texture_grad(mat.orm_texture_idx, uv, uv_ddx, uv_ddy, mat.flags)).rgb;
	float roughness = saturate(orm.g * mat.roughness);
	float metalness = saturate(orm.b * mat.metallic);

	// BaseMaterial3D: (emission + texture) or (emission * texture), times the energy; the texture defaults to black.
	vec3 emissive = vec3(0.0);
	if ((mat.flags & MAT_FLAG_EMISSION) != 0u) {
		vec3 emission_tex = vec3(0.0);
		if ((mat.flags & 2u) != 0u) {
			emission_tex = (triplanar ? sample_material_triplanar(mat.emission_texture_idx, triplanar_weights, triplanar_pos, triplanar_ddx, triplanar_ddy, mat.flags) : sample_material_texture_grad(mat.emission_texture_idx, uv, uv_ddx, uv_ddy, mat.flags)).rgb;
		}
		emissive = ((mat.flags & MAT_FLAG_EMISSION_ADD) != 0u ? mat.emission_color + emission_tex : mat.emission_color * emission_tex) * mat.emission_strength;
		emissive *= scene_data_block.data.emissive_exposure_normalization;
	}

	// Build MaterialResult.
	MaterialResult m;
	m.albedo = albedo;
	m.alpha = albedo_tex.a * mat.albedo_color.a;
	m.roughness = roughness;
	m.metalness = metalness;
	m.specular = mat.specular;
	m.emissive = emissive;
	m.normal = final_normal;
	m.unshaded = (mat.flags & MAT_FLAG_UNSHADED) != 0u;
	m.ao = 1.0;
	m.ao_light_affect = 0.0;
	if ((mat.flags & MAT_FLAG_AO_TEXTURE) != 0u) {
		vec4 ao_tex = triplanar ? sample_material_triplanar(mat.ao_texture_idx, triplanar_weights, triplanar_pos, triplanar_ddx, triplanar_ddy, mat.flags) : sample_material_texture_grad(mat.ao_texture_idx, uv, uv_ddx, uv_ddy, mat.flags);
		m.ao = clamp(ao_tex[mat.ao_channel], 0.0, 1.0);
		m.ao_light_affect = mat.ao_light_affect;
	} else if ((mat.flags & MAT_FLAG_AO_ORM) != 0u) {
		m.ao = clamp(orm.r, 0.0, 1.0);
		m.ao_light_affect = mat.ao_light_affect;
	}
	m.backlight = vec3(0.0);
	if ((mat.flags & MAT_FLAG_BACKLIGHT) != 0u) {
		m.backlight = mat.backlight;
		if ((mat.flags & MAT_FLAG_HAS_BACKLIGHT_TEX) != 0u) {
			m.backlight += (triplanar ? sample_material_triplanar(mat.backlight_texture_idx, triplanar_weights, triplanar_pos, triplanar_ddx, triplanar_ddy, mat.flags) : sample_material_texture_grad(mat.backlight_texture_idx, uv, uv_ddx, uv_ddy, mat.flags)).rgb;
		}
	}
	m.specular_disabled = (mat.flags & MAT_FLAG_SPECULAR_DISABLED) != 0u;

#ifdef RT_DEBUG_ENABLED
	{
		int VIS_MODE = int(get_rt_param(RT_PARAM_VIS_MODE));
		if (VIS_MODE != 0) {
			vec3 V = -gl_WorldRayDirectionEXT;
			float NdotV = max(dot(m.normal, V), 0.0001);
			debug_visualize(VIS_MODE, h.geometry_normal, final_normal, tangent_space_normal,
					h.tangent, h.bitangent, uv, albedo, orm, metalness, roughness, mat.specular, emissive, V, NdotV);
			return;
		}
	}
#endif // RT_DEBUG_ENABLED
	shade_and_bounce(h, m);
#endif
}

#[any_hit]

#version 460

#VERSION_DEFINES

#pragma shader_stage(any_hit)
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

#define GLSL 1
#define RT_STAGE_ANY_HIT 1

// clang-format off
#include "raytracing_inc.glsl"
#include "../scene_data_inc.glsl"
#include "raytracing_common_inc.glsl"
// clang-format on

#define attribs hit_attribs.bary_or_uv
#define RT_HIT_ATTRIBS_DECLARED

#include "raytracing_hit_inc.glsl"

layout(location = 0) rayPayloadInEXT PathPayload payload;

// clang-format off
layout(set = 0, binding = 3, std430) readonly buffer GeometryBuffer {
	GeometryData geometries[];
};

layout(set = 0, binding = 4, std430) readonly buffer MotionIndexBuffer {
	int motion_indices[];
};

layout(set = 0, binding = 5, std430) readonly buffer MaterialBuffer {
	MaterialData materials[];
};
// clang-format on

layout(set = 1, binding = 0) uniform texture2D bindless_textures[];
layout(set = 2, binding = 0) uniform texture2DArray bindless_texture_arrays[];

#include "raytracing_samplers_inc.glsl"

// clang-format off
layout(set = 0, binding = 32, std430) readonly buffer MotionTransforms {
	InstanceMotionData motion_transforms[];
};
// clang-format on

// ============================================================================
// CUSTOM SHADER GLOBALS (injected for per-HG any-hit)
// ============================================================================
#ifdef RT_CUSTOM_HIT_GROUP
#include "raytracing_custom_globals_inc.glsl"
#endif

void main() {
	uint geometry_idx = gl_InstanceCustomIndexEXT;
	GeometryData geom = geometries[geometry_idx];
	if (geometry_idx == 0xFFFFFFFFu && textureLod(sampler2DArray(bindless_texture_arrays[nonuniformEXT(geometry_idx)], SAMPLER_LINEAR_REPEAT), vec3(0.0), 0.0).r > 0.0) {
		ignoreIntersectionEXT;
	}

	uint i0, i1, i2;
	get_triangle_indices(geom, i0, i1, i2);
	vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

#ifdef RT_CUSTOM_HIT_GROUP
	// Compute hit data inline (cannot include closest_hit_common_inc here).
	uint rt_geometry_idx = geometry_idx;
	vec2 rt_uv = fetch_uv(geom, i0, i1, i2, bary);
	TBNResult ah_tbn = fetch_tbn(geom, i0, i1, i2, bary);

	mat3 model_rotation = mat3(gl_ObjectToWorldEXT);
	mat3 normal_matrix = mat3(
			normalize(model_rotation[0]),
			normalize(model_rotation[1]),
			normalize(model_rotation[2]));

	vec3 rt_normal = normalize(normal_matrix * ah_tbn.normal);
	vec3 rt_tangent = normalize(normal_matrix * ah_tbn.tangent);
	vec3 rt_bitangent = cross(rt_normal, rt_tangent) * ah_tbn.bitangent_sign;

	bool rt_front_face = (gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT);
	if (!rt_front_face) {
		rt_normal = -rt_normal;
	}

	vec3 rt_hit_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	vec4 rt_color = fetch_color(geom, i0, i1, i2, bary);
	uint rt_i0 = i0;
	uint rt_i1 = i1;
	uint rt_i2 = i2;
	vec3 rt_bary = bary;

#define RT_CUSTOM_REMOVED_POINT ignoreIntersectionEXT;
#include "raytracing_custom_fragment_inc.glsl"

	if (alpha_scissor_threshold > 0.0 && alpha < alpha_scissor_threshold) {
		ignoreIntersectionEXT;
	}
#else
	// HG0: Standard material alpha test.
	vec2 uv = fetch_uv(geom, i0, i1, i2, bary);
	MaterialData mat = materials[geometry_idx];
	uv = uv * mat.uv1_scale + mat.uv1_offset;
	// The mip level the closest hit and raster read, so the cut-out does not change from one pixel's ray to the next.
	vec3 bary_ddx, bary_ddy;
	hit_barycentric_derivatives(geom, i0, i1, i2, bary, bary_ddx, bary_ddy);
	vec2 uv0 = fetch_uv(geom, i0, i0, i0, vec3(1.0, 0.0, 0.0));
	vec2 uv1 = fetch_uv(geom, i1, i1, i1, vec3(1.0, 0.0, 0.0));
	vec2 uv2 = fetch_uv(geom, i2, i2, i2, vec3(1.0, 0.0, 0.0));
	vec2 uv_ddx = (uv0 * bary_ddx.x + uv1 * bary_ddx.y + uv2 * bary_ddx.z) * mat.uv1_scale;
	vec2 uv_ddy = (uv0 * bary_ddy.x + uv1 * bary_ddy.y + uv2 * bary_ddy.z) * mat.uv1_scale;
	float alpha = textureGrad(sampler2D(bindless_textures[nonuniformEXT(mat.albedo_texture_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), uv, uv_ddx, uv_ddy).a;
	alpha *= mat.albedo_color.a;
	if ((mat.flags & MAT_FLAG_VERTEX_COLOR_ALBEDO) != 0u) {
		alpha *= fetch_color(geom, i0, i1, i2, bary).a;
	}

	if (alpha < 0.5) {
		ignoreIntersectionEXT;
	}
#endif
}

#[intersection]

#version 460

#VERSION_DEFINES

#pragma shader_stage(intersection)
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

#define GLSL 1
#define RT_STAGE_INTERSECTION 1

// clang-format off
#include "raytracing_inc.glsl"
#include "../scene_data_inc.glsl"
#include "raytracing_data_inc.glsl"
#include "raytracing_common_inc.glsl"
// clang-format on

// Write all attributes and report the intersection. Transparently delta-compresses
// PREV_POSITION into spare .w bytes of packed_normal/tangent + prev_pos_delta_yz.
#define report_intersection(t_hit, kind)                                                      \
	{                                                                                         \
		vec3 _obj_hit = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * (t_hit);           \
		vec3 _delta = any(isnan(m_PREV_POSITION)) ? vec3(0.0) : (m_PREV_POSITION - _obj_hit); \
		uint _n4 = packSnorm4x8(vec4(m_HIT_NORMAL, 0.0));                                     \
		uint _t4 = packSnorm4x8(vec4(m_HIT_TANGENT, 0.0));                                    \
		uint _dx = packHalf2x16(vec2(_delta.x, 0.0));                                         \
		hit_attribs.bary_or_uv = m_HIT_UV;                                                    \
		hit_attribs.packed_normal = (_n4 & 0x00FFFFFFu) | ((_dx & 0xFFu) << 24u);             \
		hit_attribs.packed_tangent = (_t4 & 0x00FFFFFFu) | (((_dx >> 8u) & 0xFFu) << 24u);    \
		hit_attribs.prev_pos_delta_yz = packHalf2x16(vec2(_delta.y, _delta.z));               \
		reportIntersectionEXT(t_hit, kind);                                                   \
	}

#ifdef RT_CUSTOM_HIT_GROUP

// clang-format off
layout(set = 0, binding = 3, std430) readonly buffer GeometryBuffer {
	GeometryData geometries[];
};

layout(set = 0, binding = 5, std430) readonly buffer MaterialBuffer {
	MaterialData materials[];
};
// clang-format on

layout(set = 1, binding = 0) uniform texture2D bindless_textures[];
layout(set = 2, binding = 0) uniform texture2DArray bindless_texture_arrays[];

#include "raytracing_samplers_inc.glsl"

layout(buffer_reference, std140) readonly buffer CustomMaterialUniforms{
	/* RT_CUSTOM_UNIFORM_MEMBERS */
};

/* RT_CUSTOM_TEXTURE_DEFINES */

// File-scope built-ins accessible from user helper functions in globals.
float global_time = scene_data_block.data.time;
float global_prev_time = 0.0;
mat4 read_model_matrix = mat4(0.0);
mat4 m_INV_MODEL_MATRIX = mat4(0.0);
mat4 read_view_matrix = transpose(mat4(scene_data_block.data.view_matrix[0], scene_data_block.data.view_matrix[1], scene_data_block.data.view_matrix[2], vec4(0.0, 0.0, 0.0, 1.0)));
mat4 inv_view_matrix = transpose(mat4(scene_data_block.data.inv_view_matrix[0], scene_data_block.data.inv_view_matrix[1], scene_data_block.data.inv_view_matrix[2], vec4(0.0, 0.0, 0.0, 1.0)));
mat4 projection_matrix = scene_data_block.data.projection_matrix;
mat4 inv_projection_matrix = scene_data_block.data.inv_projection_matrix;
vec2 read_viewport_size = scene_data_block.data.viewport_size;
float m_Z_NEAR = scene_data_block.data.z_near;
float m_Z_FAR = scene_data_block.data.z_far;

uint64_t _rt_material_address;
#define material CustomMaterialUniforms(_rt_material_address)

/* RT_CUSTOM_INTERSECTION_GLOBALS */

#endif

void main() {
	if (gl_InstanceCustomIndexEXT == -1 && textureLod(sampler2DArray(bindless_texture_arrays[nonuniformEXT(0)], SAMPLER_LINEAR_REPEAT), vec3(0.0), 0.0).r > 0.0) {
		return;
	}
#ifdef RT_CUSTOM_HIT_GROUP
	// Writable outputs.
	vec2 m_HIT_UV = vec2(0.0);
	vec3 m_HIT_NORMAL = vec3(0.0, 1.0, 0.0);
	vec3 m_HIT_TANGENT = vec3(1.0, 0.0, 0.0);
	vec3 m_PREV_POSITION = vec3(uintBitsToFloat(0x7FC00000u)); // NaN sentinel = not set.

	// Per-invocation built-ins (require RT intrinsics, only available in main).
	vec3 m_ORIGIN = gl_ObjectRayOriginEXT;
	vec3 m_DIRECTION = gl_ObjectRayDirectionEXT;
	vec3 m_WORLD_ORIGIN = gl_WorldRayOriginEXT;
	vec3 m_WORLD_DIRECTION = gl_WorldRayDirectionEXT;
	float m_T_MIN = gl_RayTminEXT;
	float m_T_MAX = gl_RayTmaxEXT;
	global_prev_time = scene_data_block.prev_data.time;

	// Resolve custom material uniforms via BDA (assigns file-scope address).
	uint rt_geometry_idx = gl_InstanceCustomIndexEXT;
	MaterialData rt_mat = materials[rt_geometry_idx];
	_rt_material_address = rt_mat.uniform_address;

	// Per-primitive AABB bounds (available when expose_aabb_bounds is enabled).
	GeometryData rt_geom = geometries[rt_geometry_idx];
	vec3 m_AABB_MIN = vec3(0.0);
	vec3 m_AABB_MAX = vec3(0.0);
	if (rt_geom.vertex_address != 0ul) {
		FloatBuffer aabb_buf = FloatBuffer(rt_geom.vertex_address);
		int base = int(gl_PrimitiveID) * 6;
		m_AABB_MIN = vec3(aabb_buf.v[base + 0], aabb_buf.v[base + 1], aabb_buf.v[base + 2]);
		m_AABB_MAX = vec3(aabb_buf.v[base + 3], aabb_buf.v[base + 4], aabb_buf.v[base + 5]);
	}

	mat4 rt_aabb_xform;
	mat4 rt_inv_aabb_xform;
	get_aabb_compression_xforms(rt_geom, rt_aabb_xform, rt_inv_aabb_xform);
	read_model_matrix = mat4(gl_ObjectToWorldEXT) * rt_inv_aabb_xform;
	m_INV_MODEL_MATRIX = rt_aabb_xform * mat4(gl_WorldToObjectEXT);

	/* RT_CUSTOM_INTERSECTION_CODE */

#else
	// Base-variant fallback: never executed at runtime (the intersection
	// stage is always rebuilt per-HG with RT_CUSTOM_HIT_GROUP defined).
	// Only touch unconditional HitAttribs fields so the base variant parses.
	hit_attribs.bary_or_uv = vec2(0.0);
	reportIntersectionEXT(gl_RayTminEXT, 0u);
#endif
}

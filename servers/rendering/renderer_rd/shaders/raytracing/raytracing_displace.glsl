#[compute]

#version 460

#VERSION_DEFINES

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

// Runs a material's vertex() on every vertex of a mesh surface, so the BLAS holds the geometry raster draws.
// The material code is injected per hit group slot, as in the hit shaders.
//
// Output, per job: [float3 displaced position x N] [rest TBN block x N, the mesh's uncompressed layout]
// [float3 rest position x N] [float3 previous frame's displaced position x N]. The first two sections are the
// deformed-path vertex buffer; the hit shaders run vertex() again on the rest data at the triangle's vertices to get
// raster's varyings; the last one gives the motion vectors.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

#define GLSL 1
#define RT_DISPLACE_PASS 1

// clang-format off
#include "raytracing_inc.glsl"
#include "../scene_data_inc.glsl"
#include "raytracing_hit_inc.glsl"
// clang-format on

layout(set = 0, binding = 2, std140) uniform SceneDataBlock {
	SceneData data;
	SceneData prev_data;
}
scene_data_block;

layout(set = 0, binding = 14, std430) readonly buffer GlobalShaderUniformData {
	vec4 data[];
}
global_shader_uniforms;

#include "raytracing_samplers_inc.glsl"

struct DisplaceJob {
	mat4 model_matrix;
	mat4 inv_model_matrix;
	mat4 prev_model_matrix; // The current one for a teleported instance, as raster's motion vectors.
	mat4 prev_inv_model_matrix;
	GeometryData geom; // The mesh surface as stored (possibly compressed).
	uint64_t material_address;
	uint tbn_words; // 0, 1 (normal) or 2 (normal and tangent) per vertex.
	uint instance_uniforms_ofs;
};

layout(set = 0, binding = 30, std430) readonly buffer DisplaceJobs {
	DisplaceJob jobs[];
};

layout(set = 1, binding = 0) uniform texture2D bindless_textures[];
layout(set = 2, binding = 0) uniform texture2DArray bindless_texture_arrays[];

layout(set = 3, binding = 0, std430) restrict writeonly buffer DisplaceOutput {
	uint v[];
}
dst;

// Bound so the graph orders the material buffer's upload before this pass; read through its address.
layout(set = 3, binding = 1, std430) restrict readonly buffer MaterialDependency {
	uint v[];
}
material_dependency;

layout(push_constant, std430) uniform Params {
	uint job_index;
	uint never; // Always 0: keeps every binding statically used, so all slots share the set layouts.
	uint pad[2];
}
params;

#include "raytracing_custom_globals_inc.glsl"

vec3 fetch_position(in GeometryData p_geom, uint p_index) {
	if ((p_geom.flags & FLAG_COMPRESSED) != 0u) {
		Uint32Buffer vb = Uint32Buffer(p_geom.vertex_address);
		uint w0 = vb.v[p_index * 2u];
		uint w1 = vb.v[p_index * 2u + 1u];
		vec3 unorm = vec3(float(w0 & 0xFFFFu), float(w0 >> 16), float(w1 & 0xFFFFu)) / 65535.0;
		return unorm * vec3(p_geom.aabb_size_x, p_geom.aabb_size_y, p_geom.aabb_size_z) + vec3(p_geom.aabb_pos_x, p_geom.aabb_pos_y, p_geom.aabb_pos_z);
	}
	FloatBuffer vb = FloatBuffer(p_geom.vertex_address);
	return vec3(vb.v[p_index * 3u], vb.v[p_index * 3u + 1u], vb.v[p_index * 3u + 2u]);
}

uint pack_unorm16x2(vec2 p_value) {
	uvec2 u = uvec2(clamp(p_value, 0.0, 1.0) * 65535.0);
	return u.x | (u.y << 16);
}

// Runs vertex() on a rest vertex with this frame's or the previous frame's data; returns the position in the mesh
// space of that frame's transform.
vec3 displace_vertex(DisplaceJob p_job, vec3 p_rest_position, TBNResult p_tbn, uint p_index, bool p_previous) {
	mat4 model = p_previous ? p_job.prev_model_matrix : p_job.model_matrix;
	mat4 inv_model = p_previous ? p_job.prev_inv_model_matrix : p_job.inv_model_matrix;
	read_model_matrix = model;
	model_normal_matrix = transpose(mat3(inv_model));
	if (p_previous) {
		read_view_matrix = transpose(mat4(scene_data_block.prev_data.view_matrix[0], scene_data_block.prev_data.view_matrix[1], scene_data_block.prev_data.view_matrix[2], vec4(0.0, 0.0, 0.0, 1.0)));
		inv_view_matrix = transpose(mat4(scene_data_block.prev_data.inv_view_matrix[0], scene_data_block.prev_data.inv_view_matrix[1], scene_data_block.prev_data.inv_view_matrix[2], vec4(0.0, 0.0, 0.0, 1.0)));
		projection_matrix = scene_data_block.prev_data.projection_matrix;
		inv_projection_matrix = scene_data_block.prev_data.inv_projection_matrix;
		read_viewport_size = scene_data_block.prev_data.viewport_size;
		global_time = scene_data_block.prev_data.time;
	} else {
		read_view_matrix = transpose(mat4(scene_data_block.data.view_matrix[0], scene_data_block.data.view_matrix[1], scene_data_block.data.view_matrix[2], vec4(0.0, 0.0, 0.0, 1.0)));
		inv_view_matrix = transpose(mat4(scene_data_block.data.inv_view_matrix[0], scene_data_block.data.inv_view_matrix[1], scene_data_block.data.inv_view_matrix[2], vec4(0.0, 0.0, 0.0, 1.0)));
		projection_matrix = scene_data_block.data.projection_matrix;
		inv_projection_matrix = scene_data_block.data.inv_projection_matrix;
		read_viewport_size = scene_data_block.data.viewport_size;
		global_time = scene_data_block.data.time;
	}
	rt_modelview_matrix = read_view_matrix * read_model_matrix;
	rt_modelview_normal_matrix = mat3(rt_modelview_matrix);

#ifdef VERTEX_WORLD_COORDS_USED
	vertex = (read_model_matrix * vec4(p_rest_position, 1.0)).xyz;
	normal = normalize(model_normal_matrix * p_tbn.normal);
	tangent = normalize(mat3(read_model_matrix) * p_tbn.tangent);
	binormal = normalize(mat3(read_model_matrix) * p_tbn.binormal);
#else
	vertex = p_rest_position;
	normal = p_tbn.normal;
	tangent = p_tbn.tangent;
	binormal = p_tbn.binormal;
#endif
	uv_interp = fetch_uv(p_job.geom, p_index, p_index, p_index, vec3(1.0, 0.0, 0.0));
	uv2_interp = fetch_uv2(p_job.geom, p_index, p_index, p_index, vec3(1.0, 0.0, 0.0));
	color_interp = fetch_color(p_job.geom, p_index, p_index, p_index, vec3(1.0, 0.0, 0.0));
	rt_vertex_id = int(p_index);

	/* RT_CUSTOM_VERTEX_CALL */

	// Back to mesh space, the space of the BLAS.
#if defined(SKIP_TRANSFORM_USED)
	vec3 displaced = (inv_model * (inv_view_matrix * vec4(vertex, 1.0))).xyz;
#elif defined(VERTEX_WORLD_COORDS_USED)
	vec3 displaced = (inv_model * vec4(vertex, 1.0)).xyz;
#elif defined(RT_MODELVIEW_WRITTEN)
	// Through the MODELVIEW_MATRIX vertex() wrote (a billboard), as raster draws it, then back from view space.
	vec3 displaced = (inv_model * (inv_view_matrix * (rt_modelview_matrix * vec4(vertex, 1.0)))).xyz;
#else
	vec3 displaced = vertex;
#endif
	// A point vertex() removed (NaN) makes its triangles inactive in the BLAS, which needs a NaN x. An infinite one too
	// (an effect dividing by a zero duration): raster clips it, but a BLAS built from it is invalid.
	if (any(isnan(displaced)) || any(isinf(displaced))) {
		displaced = vec3(uintBitsToFloat(0x7fc00000u));
	}
	return displaced;
}

void main() {
	DisplaceJob job = jobs[params.job_index];
	uint index = gl_GlobalInvocationID.x;

	if (params.never != 0u) {
		float value = global_shader_uniforms.data[0].x + scene_data_block.data.time;
		value += texelFetch(sampler2D(bindless_textures[nonuniformEXT(index)], SAMPLER_NEAREST_CLAMP), ivec2(0), 0).r;
		value += texelFetch(sampler2DArray(bindless_texture_arrays[nonuniformEXT(index)], SAMPLER_NEAREST_CLAMP), ivec3(0), 0).r;
		dst.v[0] = material_dependency.v[0] + floatBitsToUint(value);
	}

	GeometryData geom = job.geom;
	if (index >= geom.vertex_count) {
		return;
	}

	// The rest vertex, decoded as the hit shaders decode it.
	vec3 rest_position = fetch_position(geom, index);
	TBNResult tbn;
	tbn.normal = vec3(0.0, 1.0, 0.0);
	tbn.tangent = vec3(1.0, 0.0, 0.0);
	tbn.binormal = vec3(0.0, 0.0, 1.0);
	tbn.bitangent_sign = 1.0;
	if (geom.normal_byte_offset != OFFSET_NONE) {
		tbn = (geom.flags & FLAG_COMPRESSED) != 0u ? fetch_tbn_compressed_vertex(geom, index) : fetch_tbn_uncompressed_vertex(geom, index);
	}

	material = CustomMaterialUniforms(job.material_address);
	rt_instance_uniforms_ofs = job.instance_uniforms_ofs;

	// As raster computes motion vectors: vertex() again with the previous frame's time, camera and transform (the
	// uniforms stay current), for the motion of the previous position.
	vec3 previous = displace_vertex(job, rest_position, tbn, index, true);
	vec3 displaced = displace_vertex(job, rest_position, tbn, index, false);

	uint vertex_count = geom.vertex_count;
	dst.v[index * 3u] = floatBitsToUint(displaced.x);
	dst.v[index * 3u + 1u] = floatBitsToUint(displaced.y);
	dst.v[index * 3u + 2u] = floatBitsToUint(displaced.z);

	uint previous_base = vertex_count * (6u + job.tbn_words) + index * 3u;
	dst.v[previous_base] = floatBitsToUint(previous.x);
	dst.v[previous_base + 1u] = floatBitsToUint(previous.y);
	dst.v[previous_base + 2u] = floatBitsToUint(previous.z);

	if (job.tbn_words > 0u) {
		uint tbn_base = vertex_count * 3u + index * job.tbn_words;
		if ((geom.flags & FLAG_COMPRESSED) == 0u) {
			// Same layout: copy the words.
			Uint32Buffer vb = Uint32Buffer(geom.vertex_address);
			uint src_base = (geom.normal_byte_offset >> 2) + index * (geom.normal_stride >> 2);
			for (uint w = 0u; w < job.tbn_words; w++) {
				dst.v[tbn_base + w] = vb.v[src_base + w];
			}
		} else {
			// Vector3::octahedron_encode() and octahedron_tangent_encode().
			dst.v[tbn_base] = pack_unorm16x2(vec3_to_oct(tbn.normal));
			if (job.tbn_words > 1u) {
				vec2 t = vec3_to_oct(tbn.tangent);
				t.y = max(t.y, 1.0 / 32767.0) * 0.5 + 0.5;
				t.y = tbn.bitangent_sign >= 0.0 ? t.y : 1.0 - t.y;
				dst.v[tbn_base + 1u] = pack_unorm16x2(t);
			}
		}
	}

	uint rest_base = vertex_count * (3u + job.tbn_words) + index * 3u;
	dst.v[rest_base] = floatBitsToUint(rest_position.x);
	dst.v[rest_base + 1u] = floatBitsToUint(rest_position.y);
	dst.v[rest_base + 2u] = floatBitsToUint(rest_position.z);
}

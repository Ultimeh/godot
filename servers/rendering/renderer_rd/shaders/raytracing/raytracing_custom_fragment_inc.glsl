// Custom shader fragment setup for RT hit groups (closest-hit and any-hit).
// Include inside main() under #ifdef RT_CUSTOM_HIT_GROUP.
//
// Required variables before inclusion:
//   uint  rt_geometry_idx   -- geometry/material index
//   vec3  rt_hit_pos        -- world-space hit position
//   uint  rt_i0, rt_i1, rt_i2 -- triangle vertex indices
//   vec3  rt_bary           -- barycentrics of the hit
//   vec2  rt_uv             -- interpolated UV
//   vec3  rt_normal         -- world-space geometry normal (flipped for back-face)
//   vec3  rt_tangent        -- world-space tangent
//   vec3  rt_bitangent      -- world-space bitangent
//   bool  rt_front_face     -- true if front-face hit
//
// Required bindings/types:
//   materials[], CustomMaterialUniforms, scene_data_block

MaterialData rt_mat = materials[rt_geometry_idx];
material = CustomMaterialUniforms(rt_mat.uniform_address);

// Matrices.
mat4 rt_view_matrix = transpose(mat4(scene_data_block.data.view_matrix[0],
		scene_data_block.data.view_matrix[1],
		scene_data_block.data.view_matrix[2],
		vec4(0.0, 0.0, 0.0, 1.0)));

GeometryData rt_geom = geometries[rt_geometry_idx];
mat4 rt_aabb_xform;
mat4 rt_inv_aabb_xform;
get_aabb_compression_xforms(rt_geom, rt_aabb_xform, rt_inv_aabb_xform);

read_model_matrix = mat4(gl_ObjectToWorldEXT) * rt_inv_aabb_xform;
model_normal_matrix = transpose(inverse(mat3(read_model_matrix)));
read_view_matrix = rt_view_matrix;
inv_view_matrix = transpose(mat4(scene_data_block.data.inv_view_matrix[0],
		scene_data_block.data.inv_view_matrix[1],
		scene_data_block.data.inv_view_matrix[2],
		vec4(0.0, 0.0, 0.0, 1.0)));
projection_matrix = scene_data_block.data.projection_matrix;
inv_projection_matrix = scene_data_block.data.inv_projection_matrix;
read_viewport_size = scene_data_block.data.viewport_size;
global_time = scene_data_block.data.time;

mat4 rt_world_to_object_decomp = rt_aabb_xform * mat4(gl_WorldToObjectEXT);

#ifdef VERTEX_WORLD_COORDS_USED
vertex = rt_hit_pos;
normal = rt_normal;
tangent = rt_tangent;
binormal = rt_bitangent;
#else
vertex = (rt_world_to_object_decomp * vec4(rt_hit_pos, 1.0)).xyz;
normal = mat3(rt_world_to_object_decomp) * rt_normal;
tangent = mat3(rt_world_to_object_decomp) * rt_tangent;
binormal = mat3(rt_world_to_object_decomp) * rt_bitangent;
#endif
uv_interp = rt_uv;
uv2_interp = rt_uv;
color_interp = rt_color;
view = -gl_WorldRayDirectionEXT;
rt_front_facing = rt_front_face;
rt_screen_uv = vec2(gl_LaunchIDEXT.xy) / vec2(gl_LaunchSizeEXT.xy);
rt_frag_coord = vec4(gl_LaunchIDEXT.xy, 0.0, 1.0);

rt_modelview_matrix = rt_view_matrix * read_model_matrix;
rt_modelview_normal_matrix = mat3(rt_modelview_matrix);

if ((rt_geom.flags & FLAG_PROCEDURAL) == 0u) {
	// As raster does, run vertex() at the triangle's vertices (on their rest data for displaced geometry, whose BLAS
	// holds vertex()'s output already) and interpolate what it writes: run once at the hit, a texelFetch() in vertex()
	// would snap to one texel. The vertex values also give the screen derivatives.
	uint rt_corner_index[3] = uint[3](rt_i0, rt_i1, rt_i2);
	for (int rt_corner = 0; rt_corner < 3; rt_corner++) {
		uint rt_index = rt_corner_index[rt_corner];
		vec3 rt_rest = (rt_geom.flags & FLAG_DISPLACED) != 0u ? fetch_rest_vertex(rt_geom, rt_index) : fetch_mesh_position(rt_geom, rt_index);
		TBNResult rt_tbn;
		rt_tbn.normal = vec3(0.0, 1.0, 0.0);
		rt_tbn.tangent = vec3(1.0, 0.0, 0.0);
		rt_tbn.binormal = vec3(0.0, 0.0, 1.0);
		if (rt_geom.normal_byte_offset != OFFSET_NONE) {
			rt_tbn = (rt_geom.flags & FLAG_COMPRESSED) != 0u ? fetch_tbn_compressed_vertex(rt_geom, rt_index) : fetch_tbn_uncompressed_vertex(rt_geom, rt_index);
		}
#ifdef VERTEX_WORLD_COORDS_USED
		vertex = (read_model_matrix * vec4(rt_rest, 1.0)).xyz;
		normal = normalize(model_normal_matrix * rt_tbn.normal);
		tangent = normalize(mat3(read_model_matrix) * rt_tbn.tangent);
		binormal = normalize(mat3(read_model_matrix) * rt_tbn.binormal);
#else
		vertex = rt_rest;
		normal = rt_tbn.normal;
		tangent = rt_tbn.tangent;
		binormal = rt_tbn.binormal;
#endif
		uv_interp = fetch_uv(rt_geom, rt_index, rt_index, rt_index, vec3(1.0, 0.0, 0.0));
		uv2_interp = uv_interp;
		color_interp = fetch_color(rt_geom, rt_index, rt_index, rt_index, vec3(1.0, 0.0, 0.0));
		rt_vertex_id = int(rt_index);

		/* RT_CUSTOM_VERTEX_CALL */

		rt_corner_vertex[rt_corner] = vertex;
		rt_corner_normal[rt_corner] = normal;
		rt_corner_tangent[rt_corner] = tangent;
		rt_corner_binormal[rt_corner] = binormal;
		rt_corner_uv[rt_corner] = uv_interp;
		rt_corner_uv2[rt_corner] = uv2_interp;
		rt_corner_color[rt_corner] = color_interp;
		/* RT_CUSTOM_VARYING_SAVE */
	}
	if (any(isnan(rt_corner_vertex[0])) || any(isnan(rt_corner_vertex[1])) || any(isnan(rt_corner_vertex[2]))) {
		RT_CUSTOM_REMOVED_POINT
	}
	rt_hit_bary = rt_bary;
	rt_set_fragment_inputs(rt_bary);
	hit_barycentric_derivatives(rt_geom, rt_i0, rt_i1, rt_i2, rt_bary, rt_hit_bary_ddx, rt_hit_bary_ddy);
	rt_derivatives_ready = true;
} else {
	// Procedural geometry has no vertices: vertex() runs once at the hit, without screen derivatives.
	/* RT_CUSTOM_VERTEX_CALL */
	if (any(isnan(vertex))) {
		RT_CUSTOM_REMOVED_POINT
	}
#if defined(SKIP_TRANSFORM_USED)
	// vertex() already wrote view-space values.
#elif defined(VERTEX_WORLD_COORDS_USED)
	vertex = (rt_view_matrix * vec4(vertex, 1.0)).xyz;
	normal = normalize(mat3(rt_view_matrix) * normal);
	tangent = normalize(mat3(rt_view_matrix) * tangent);
	binormal = normalize(mat3(rt_view_matrix) * binormal);
#else
	vertex = (rt_modelview_matrix * vec4(vertex, 1.0)).xyz;
	normal = normalize(rt_modelview_normal_matrix * normal);
	tangent = normalize(mat3(rt_modelview_matrix) * tangent);
	binormal = normalize(mat3(rt_modelview_matrix) * binormal);
#endif
}

// Fragment outputs with sensible defaults.
vec3 albedo = vec3(1.0);
float alpha = 1.0;
float metallic = 0.0;
float roughness = 0.5;
float specular = 0.5;
vec3 emission = vec3(0.0);
vec3 normal_map = vec3(0.5, 0.5, 1.0);
float normal_map_depth = 1.0;
float ao = 1.0;
float ao_light_affect = 0.0;
vec3 backlight = vec3(0.0);
float sss_strength = 0.0;
float rim = 0.0;
float rim_tint = 0.0;
float clearcoat = 0.0;
float clearcoat_roughness = 0.0;
float anisotropy = 0.0;
vec2 anisotropy_flow = vec2(1.0, 0.0);
float alpha_scissor_threshold = 0.0;
float alpha_hash_scale = 1.0;
float alpha_antialiasing_edge = 0.0;
vec2 alpha_texture_coordinate = vec2(0.0);

{
	/* RT_CUSTOM_FRAGMENT_CODE */
}

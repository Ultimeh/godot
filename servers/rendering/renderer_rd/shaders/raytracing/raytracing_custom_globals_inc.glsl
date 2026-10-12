// Custom shader globals for RT hit groups (closest-hit and any-hit).
// Include under #ifdef RT_CUSTOM_HIT_GROUP, outside main().
// Requires: bindless_textures[], materials[] bindings already declared.

/* RT_CUSTOM_DEFINES */

// Screen derivatives: the hit shaders evaluate the argument again with the varyings and vertex outputs of the
// neighboring pixel, interpolated from the triangle's vertices as raster interpolates them (rt_push_neighbor()).
// Outside the fragment code (vertex(), the displacement pass) they are 0, as raster's vertex() reads mip 0.
int rt_push_neighbor(int p_axis);
int rt_pop_neighbor();
float rt_difference(float p_neighbor, float p_value) {
	return p_neighbor - p_value;
}
vec2 rt_difference(vec2 p_neighbor, vec2 p_value) {
	return p_neighbor - p_value;
}
vec3 rt_difference(vec3 p_neighbor, vec3 p_value) {
	return p_neighbor - p_value;
}
vec4 rt_difference(vec4 p_neighbor, vec4 p_value) {
	return p_neighbor - p_value;
}
// Reads the value where it stands: a variable alone after a comma would be read only at the call, after the restore.
float rt_value(float p_value) {
	return p_value;
}
vec2 rt_value(vec2 p_value) {
	return p_value;
}
vec3 rt_value(vec3 p_value) {
	return p_value;
}
vec4 rt_value(vec4 p_value) {
	return p_value;
}
// Function arguments and the comma operator are evaluated left to right.
#define dFdx(m_value) rt_difference((rt_push_neighbor(0), rt_value(m_value)), (rt_pop_neighbor(), rt_value(m_value)))
#define dFdy(m_value) rt_difference((rt_push_neighbor(1), rt_value(m_value)), (rt_pop_neighbor(), rt_value(m_value)))
#define dFdxCoarse(m_value) dFdx(m_value)
#define dFdyCoarse(m_value) dFdy(m_value)
#define dFdxFine(m_value) dFdx(m_value)
#define dFdyFine(m_value) dFdy(m_value)
#define fwidth(m_value) (abs(dFdx(m_value)) + abs(dFdy(m_value)))
#define fwidthCoarse(m_value) fwidth(m_value)
#define fwidthFine(m_value) fwidth(m_value)
#define SHADER_IS_SRGB false
#define SHADER_SPACE_FAR 0.0

layout(buffer_reference, std140) readonly buffer CustomMaterialUniforms{
	/* RT_CUSTOM_UNIFORM_MEMBERS */
};

// Shared vertex/fragment built-ins at global scope so that the vertex function,
// texture #defines, and fragment_globals functions can all access them.
// Assigned to real hit values in main() before use.
CustomMaterialUniforms material = CustomMaterialUniforms(uint64_t(0));
uint rt_instance_uniforms_ofs = 0u; // `instance uniform` values: global_shader_uniforms.data[rt_instance_uniforms_ofs + n].
vec3 vertex = vec3(0.0);
vec3 normal = vec3(0.0, 0.0, 1.0);
vec3 tangent = vec3(1.0, 0.0, 0.0);
vec3 binormal = vec3(0.0, 1.0, 0.0);
vec2 uv_interp = vec2(0.0);
vec2 uv2_interp = vec2(0.0);
vec4 color_interp = vec4(1.0);
vec3 view = vec3(0.0, 0.0, -1.0);
mat4 read_model_matrix = mat4(1.0);
mat3 model_normal_matrix = mat3(1.0);
mat4 read_view_matrix = mat4(1.0);
mat4 rt_modelview_matrix = mat4(1.0);
mat3 rt_modelview_normal_matrix = mat3(1.0);
mat4 inv_view_matrix = mat4(1.0);
mat4 projection_matrix = mat4(1.0);
mat4 inv_projection_matrix = mat4(1.0);
float global_time = 0.0;
float global_prev_time = 0.0;
int rt_vertex_id = 0;
int rt_instance_id = 0;
float rt_point_size = 1.0;
vec4 position = vec4(0.0);
vec4 instance_custom = vec4(0.0);
vec4 custom0_attrib = vec4(0.0);
vec4 custom1_attrib = vec4(0.0);
vec4 custom2_attrib = vec4(0.0);
vec4 custom3_attrib = vec4(0.0);
uvec4 bone_attrib = uvec4(0);
vec4 weight_attrib = vec4(0.0);
vec2 read_viewport_size = vec2(1.0);
bool rt_front_facing = true;
vec2 rt_screen_uv = vec2(0.0);
vec4 rt_frag_coord = vec4(0.0);
float alpha_antialiasing_edge = 0.0;
vec2 alpha_texture_coordinate = vec2(0.0);

// What vertex() wrote at the triangle's three vertices, and the hit's barycentrics with their screen derivatives.
vec3 rt_corner_vertex[3];
vec3 rt_corner_normal[3];
vec3 rt_corner_tangent[3];
vec3 rt_corner_binormal[3];
vec2 rt_corner_uv[3];
vec2 rt_corner_uv2[3];
vec4 rt_corner_color[3];
vec3 rt_hit_bary = vec3(1.0, 0.0, 0.0);
vec3 rt_hit_bary_ddx = vec3(0.0);
vec3 rt_hit_bary_ddy = vec3(0.0);
bool rt_derivatives_ready = false;

// DEPTH_TEXTURE as raster reads it at this pixel. For a front layer ray (a transparent surface's reflection) it is the
// opaque surface behind, which the raygen traced: material code that measures the water's depth, foam or shore from it
// works as in raster. Elsewhere there is nothing behind: the far plane (0, reverse Z).
float rt_behind_depth = 0.0;
float rt_depth_buffer_value() {
	return rt_behind_depth;
}

// Screen/depth textures are unavailable in RT -- alias to bindless slot 0
// so shaders that reference them still compile (reads return dummy values).
#define depth_buffer bindless_textures[0]
#define color_buffer bindless_textures[0]
#define normal_roughness_buffer bindless_textures[0]

/* RT_CUSTOM_TEXTURE_DEFINES */
/* RT_CUSTOM_FRAGMENT_GLOBALS */

/* RT_CUSTOM_VERTEX_FUNCTION */

/* RT_CUSTOM_VARYING_STORAGE */

// The fragment inputs at barycentrics p_bary: vertex()'s outputs interpolated, then in view space as raster has them.
void rt_set_fragment_inputs(vec3 p_bary) {
	vertex = rt_corner_vertex[0] * p_bary.x + rt_corner_vertex[1] * p_bary.y + rt_corner_vertex[2] * p_bary.z;
	normal = rt_corner_normal[0] * p_bary.x + rt_corner_normal[1] * p_bary.y + rt_corner_normal[2] * p_bary.z;
	tangent = rt_corner_tangent[0] * p_bary.x + rt_corner_tangent[1] * p_bary.y + rt_corner_tangent[2] * p_bary.z;
	binormal = rt_corner_binormal[0] * p_bary.x + rt_corner_binormal[1] * p_bary.y + rt_corner_binormal[2] * p_bary.z;
	uv_interp = rt_corner_uv[0] * p_bary.x + rt_corner_uv[1] * p_bary.y + rt_corner_uv[2] * p_bary.z;
	uv2_interp = rt_corner_uv2[0] * p_bary.x + rt_corner_uv2[1] * p_bary.y + rt_corner_uv2[2] * p_bary.z;
	color_interp = rt_corner_color[0] * p_bary.x + rt_corner_color[1] * p_bary.y + rt_corner_color[2] * p_bary.z;
	/* RT_CUSTOM_VARYING_BLEND */
#if defined(SKIP_TRANSFORM_USED)
	// vertex() already wrote view-space values.
#elif defined(VERTEX_WORLD_COORDS_USED)
	vertex = (read_view_matrix * vec4(vertex, 1.0)).xyz;
	normal = normalize(mat3(read_view_matrix) * normal);
	tangent = normalize(mat3(read_view_matrix) * tangent);
	binormal = normalize(mat3(read_view_matrix) * binormal);
#else
	vertex = (rt_modelview_matrix * vec4(vertex, 1.0)).xyz;
	normal = normalize(rt_modelview_normal_matrix * normal);
	tangent = normalize(mat3(rt_modelview_matrix) * tangent);
	binormal = normalize(mat3(rt_modelview_matrix) * binormal);
#endif
}

vec3 rt_saved_vertex;
vec3 rt_saved_normal;
vec3 rt_saved_tangent;
vec3 rt_saved_binormal;
vec2 rt_saved_uv;
vec2 rt_saved_uv2;
vec4 rt_saved_color;

int rt_push_neighbor(int p_axis) {
	if (!rt_derivatives_ready) {
		return 0;
	}
	rt_saved_vertex = vertex;
	rt_saved_normal = normal;
	rt_saved_tangent = tangent;
	rt_saved_binormal = binormal;
	rt_saved_uv = uv_interp;
	rt_saved_uv2 = uv2_interp;
	rt_saved_color = color_interp;
	/* RT_CUSTOM_VARYING_PUSH */
	rt_set_fragment_inputs(rt_hit_bary + (p_axis == 0 ? rt_hit_bary_ddx : rt_hit_bary_ddy));
	return 0;
}

int rt_pop_neighbor() {
	if (!rt_derivatives_ready) {
		return 0;
	}
	vertex = rt_saved_vertex;
	normal = rt_saved_normal;
	tangent = rt_saved_tangent;
	binormal = rt_saved_binormal;
	uv_interp = rt_saved_uv;
	uv2_interp = rt_saved_uv2;
	color_interp = rt_saved_color;
	/* RT_CUSTOM_VARYING_POP */
	return 0;
}

// Custom shader globals for RT hit groups (closest-hit and any-hit).
// Include under #ifdef RT_CUSTOM_HIT_GROUP, outside main().
// Requires: bindless_textures[], materials[] bindings already declared.

/* RT_CUSTOM_DEFINES */

// Ray tracing stages have no screen derivatives: material code reads mip 0.
#define dFdx(m_value) ((m_value) * 0.0)
#define dFdy(m_value) ((m_value) * 0.0)
#define dFdxCoarse(m_value) ((m_value) * 0.0)
#define dFdyCoarse(m_value) ((m_value) * 0.0)
#define dFdxFine(m_value) ((m_value) * 0.0)
#define dFdyFine(m_value) ((m_value) * 0.0)
#define fwidth(m_value) ((m_value) * 0.0)
#define fwidthCoarse(m_value) ((m_value) * 0.0)
#define fwidthFine(m_value) ((m_value) * 0.0)
#define SHADER_IS_SRGB false

layout(buffer_reference, std140) readonly buffer CustomMaterialUniforms{
	/* RT_CUSTOM_UNIFORM_MEMBERS */
};

// Shared vertex/fragment built-ins at global scope so that the vertex function,
// texture #defines, and fragment_globals functions can all access them.
// Assigned to real hit values in main() before use.
CustomMaterialUniforms material = CustomMaterialUniforms(uint64_t(0));
vec3 vertex = vec3(0.0);
vec3 normal = vec3(0.0, 0.0, 1.0);
vec3 tangent = vec3(1.0, 0.0, 0.0);
vec3 binormal = vec3(0.0, 1.0, 0.0);
vec2 uv_interp = vec2(0.0);
vec2 uv2_interp = vec2(0.0);
vec4 color_interp = vec4(1.0);
vec3 view = vec3(0.0, 0.0, -1.0);
mat4 read_model_matrix = mat4(1.0);
mat4 read_view_matrix = mat4(1.0);
mat4 rt_modelview_matrix = mat4(1.0);
mat3 rt_modelview_normal_matrix = mat3(1.0);
mat4 inv_view_matrix = mat4(1.0);
mat4 projection_matrix = mat4(1.0);
mat4 inv_projection_matrix = mat4(1.0);
float global_time = 0.0;
vec2 read_viewport_size = vec2(1.0);
bool rt_front_facing = true;
vec2 rt_screen_uv = vec2(0.0);
vec4 rt_frag_coord = vec4(0.0);
float alpha_antialiasing_edge = 0.0;
vec2 alpha_texture_coordinate = vec2(0.0);

// Screen/depth textures are unavailable in RT -- alias to bindless slot 0
// so shaders that reference them still compile (reads return dummy values).
#define depth_buffer bindless_textures[0]
#define color_buffer bindless_textures[0]
#define normal_roughness_buffer bindless_textures[0]

/* RT_CUSTOM_TEXTURE_DEFINES */
/* RT_CUSTOM_FRAGMENT_GLOBALS */

/* RT_CUSTOM_VERTEX_FUNCTION */

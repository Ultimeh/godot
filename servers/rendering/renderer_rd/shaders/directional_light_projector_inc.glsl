// The projector of a directional light (DirectionalLightData::projector_enabled): a texture over the light's XY plane,
// projected along the light, repeated beyond its size. Its color multiplies the light, rgb * a, as omni and spot
// projectors do. Include after the declaration of texture2D directional_light_projector and of the sampler
// DIRECTIONAL_LIGHT_PROJECTOR_SAMPLER (repeating).

#ifndef DIRECTIONAL_LIGHT_PROJECTOR_SAMPLER
#define DIRECTIONAL_LIGHT_PROJECTOR_SAMPLER SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT
#endif

vec2 directional_light_projector_uv(vec4 p_uv_x, vec4 p_uv_y, vec3 p_vertex) {
	return vec2(dot(p_uv_x, vec4(p_vertex, 1.0)), dot(p_uv_y, vec4(p_vertex, 1.0)));
}

// p_vertex and its screen derivatives in the space p_uv_x and p_uv_y map from.
vec3 directional_light_projector_color(vec4 p_uv_x, vec4 p_uv_y, vec3 p_vertex, vec3 p_vertex_ddx, vec3 p_vertex_ddy) {
	vec2 uv = directional_light_projector_uv(p_uv_x, p_uv_y, p_vertex);
	vec2 uv_ddx = vec2(dot(p_uv_x.xyz, p_vertex_ddx), dot(p_uv_y.xyz, p_vertex_ddx));
	vec2 uv_ddy = vec2(dot(p_uv_x.xyz, p_vertex_ddy), dot(p_uv_y.xyz, p_vertex_ddy));
	vec4 color = textureGrad(sampler2D(directional_light_projector, DIRECTIONAL_LIGHT_PROJECTOR_SAMPLER), uv, uv_ddx, uv_ddy);
	return color.rgb * color.a;
}

// Without derivatives (vertex lighting, volumetric fog): the top mip level.
vec3 directional_light_projector_color_lod(vec4 p_uv_x, vec4 p_uv_y, vec3 p_vertex) {
	vec4 color = textureLod(sampler2D(directional_light_projector, DIRECTIONAL_LIGHT_PROJECTOR_SAMPLER), directional_light_projector_uv(p_uv_x, p_uv_y, p_vertex), 0.0);
	return color.rgb * color.a;
}

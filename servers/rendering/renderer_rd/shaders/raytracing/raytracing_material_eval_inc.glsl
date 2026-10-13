// Material evaluation result - common output format for all hit groups.
// Standard materials and custom ShaderMaterials both produce this struct.

struct MaterialResult {
	vec3 albedo;
	float alpha;
	float roughness;
	float metalness;
	float specular; // Dielectric specular reflectance control [0..1], default 0.5 -> F0 = 0.04.
	vec3 emissive;
	vec3 normal; // Final shading normal (world space, after normal mapping).
	bool unshaded; // The albedo as it is, unlit; the path ends there.
	float ao; // The material's AO: occlusion finer than the traced geometry, on the indirect diffuse light as raster.
	float ao_light_affect; // How much of it the direct light gets too (AO_LIGHT_AFFECT).
	vec3 backlight; // BACKLIGHT: the direct light that goes through the surface, as raster's light_compute() adds it.
	bool specular_disabled; // render_mode specular_disabled: no specular from the lights; reflections stay, as in raster.
};

/// Sensible default for a mid-grey diffuse surface.
MaterialResult default_material_result(vec3 geometry_normal) {
	MaterialResult r;
	r.albedo = vec3(0.8);
	r.alpha = 1.0;
	r.roughness = 1.0; // BaseMaterial3D's default, as raster.
	r.metalness = 0.0;
	r.specular = 0.5;
	r.emissive = vec3(0.0);
	r.normal = geometry_normal;
	r.unshaded = false;
	r.ao = 1.0;
	r.ao_light_affect = 0.0;
	r.backlight = vec3(0.0);
	r.specular_disabled = false;
	return r;
}

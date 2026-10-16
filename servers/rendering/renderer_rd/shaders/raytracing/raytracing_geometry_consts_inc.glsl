// GeometryData's constants (raytracing_data_inc.glsl), apart for the raster shaders that read the path tracer's geometry.

const uint OFFSET_NONE = 0xFFFFFFFFu;
const uint CUSTOM_FORMAT_PRESENT = 8u; // In GeometryData.custom_formats, with the mesh's format.
const uint FLAG_COMPRESSED = 1u;
const uint FLAG_PROCEDURAL = 2u;
// Set when the BLAS uses a per-frame-deformed vertex buffer.
const uint FLAG_DEFORMED = 4u;
// Deformed by the material's vertex(): the rest positions are kept for the hit shaders.
const uint FLAG_DISPLACED = 8u;
// Uncompressed positions over a compressed attribute buffer (deformed copies of compressed meshes).
const uint FLAG_COMPRESSED_ATTRIBUTES = 16u;

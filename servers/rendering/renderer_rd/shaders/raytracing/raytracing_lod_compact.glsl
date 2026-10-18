#[compute]

#version 450

#VERSION_DEFINES

#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

// Reduces a mesh LOD to the vertices its indices use, for the merged MultiMesh BLASes (RTLodView), in three dispatches:
// MODE_MARK flags the vertices the LOD's indices reach, MODE_ASSIGN gives each one a place in the kept list (in no
// particular order), MODE_REMAP rewrites the indices to those places.

layout(buffer_reference, std430) readonly buffer SrcUintBuf {
	uint v[];
};

// Per source vertex: 0 when unused, then (MODE_ASSIGN on) its index in the kept list.
layout(set = 0, binding = 0, std430) restrict buffer Kept {
	uint v[];
}
kept;

layout(set = 0, binding = 1, std430) restrict buffer Counter {
	uint count;
}
counter;

// Per kept vertex: its index in the source vertex buffer.
layout(set = 0, binding = 2, std430) restrict writeonly buffer Remap {
	uint v[];
}
remap;

layout(set = 0, binding = 3, std430) restrict writeonly buffer Indices {
	uint v[];
}
indices;

layout(push_constant, std430) uniform Params {
	uint src_idx_lo; // The LOD's index buffer (immutable mesh data).
	uint src_idx_hi;
	uint index_count;
	uint src_is_16bit;
	uint vertex_count; // Of the source mesh.
	uint never; // Always 0: keeps every binding statically used, so the three modes share one set layout.
	uint pad[2];
}
params;

uint read_index(uint p_position) {
	SrcUintBuf src = SrcUintBuf(packUint2x32(uvec2(params.src_idx_lo, params.src_idx_hi)));
	if (params.src_is_16bit != 0u) {
		uint word = src.v[p_position >> 1u];
		return (p_position & 1u) != 0u ? (word >> 16u) : (word & 0xFFFFu);
	}
	return src.v[p_position];
}

void main() {
	uint i = gl_GlobalInvocationID.x;

	if (params.never != 0u) {
		kept.v[0] = counter.count;
		remap.v[0] = 0u;
		indices.v[0] = 0u;
	}

#ifdef MODE_MARK
	if (i < params.index_count) {
		uint src = read_index(i);
		if (src < params.vertex_count) {
			kept.v[src] = 1u;
		}
	}
#endif

#ifdef MODE_ASSIGN
	if (i < params.vertex_count && kept.v[i] != 0u) {
		uint slot = atomicAdd(counter.count, 1u);
		kept.v[i] = slot;
		remap.v[slot] = i;
	}
#endif

#ifdef MODE_REMAP
	if (i < params.index_count) {
		uint src = read_index(i);
		indices.v[i] = src < params.vertex_count ? kept.v[src] : 0u;
	}
#endif
}

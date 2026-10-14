#version 330

// Instanced cube faces for the LUT-shaded mode: per-vertex local face normal, rotated
// into the world by the instance orientation (a box's face normals are unaffected by
// its half-extents, so the rotation alone is enough).
in vec3 in_vert;
in vec3 in_nrm;
in vec3 in_pos;
in vec3 in_he;
in vec3 in_R0;
in vec3 in_R1;
in vec3 in_R2;
in vec3 in_tint;            // per-instance colour multiplier (1,1,1 = grey LUT as is)

uniform mat4 u_mvp;
uniform vec3 u_offset;      // per-copy world offset (periodic tiling); 0 = one copy

flat out vec3 v_nrm;
flat out vec3 v_tint;

void main() {
    mat3 R = mat3(in_R0, in_R1, in_R2);
    vec3 world = R * (in_vert * in_he) + in_pos + u_offset;
    v_nrm = R * in_nrm;
    v_tint = in_tint;
    gl_Position = u_mvp * vec4(world, 1.0);
}

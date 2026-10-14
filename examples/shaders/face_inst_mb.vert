#version 330

// LUT-shaded faces + per-vertex screen motion for the velocity-buffer motion blur.
// Motion of a surface point is exact for a rigid body: v + omega x r, r from the centre.
in vec3 in_vert;
in vec3 in_nrm;
in vec3 in_pos;
in vec3 in_he;
in vec3 in_R0;
in vec3 in_R1;
in vec3 in_R2;
in vec3 in_tint;            // per-instance colour multiplier (1,1,1 = grey LUT as is)
in vec3 in_vel;
in vec3 in_omg;

uniform mat4 u_mvp;
uniform vec3 u_offset;      // per-copy world offset (periodic tiling); 0 = one copy
uniform float u_shutter;      // seconds the shutter is open

flat out vec3 v_nrm;
flat out vec3 v_tint;
out vec4 v_cur;
out vec4 v_prev;

void main() {
    mat3 R = mat3(in_R0, in_R1, in_R2);
    vec3 r = R * (in_vert * in_he);
    vec3 world = r + in_pos + u_offset;
    vec3 vel = in_vel + cross(in_omg, r);
    v_cur = u_mvp * vec4(world, 1.0);
    v_prev = u_mvp * vec4(world - vel * u_shutter, 1.0);
    v_nrm = R * in_nrm;
    v_tint = in_tint;
    gl_Position = v_cur;
}

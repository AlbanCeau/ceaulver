#version 330

in vec3 in_vert;
in vec3 in_pos;
in vec3 in_he;
in vec3 in_R0;
in vec3 in_R1;
in vec3 in_R2;

uniform mat4 u_mvp;
uniform vec3 u_offset;      // per-copy world offset (periodic tiling); 0 = one copy
uniform float u_bias;

void main() {
    vec3 local = in_vert * in_he;
    vec3 world = mat3(in_R0, in_R1, in_R2) * local + in_pos + u_offset;
    vec4 pos = u_mvp * vec4(world, 1.0);
    pos.z -= u_bias;
    gl_Position = pos;
}

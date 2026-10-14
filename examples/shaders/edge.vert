#version 330

in vec3 in_vert;
uniform mat4 u_mvp;
uniform float u_bias;

void main() {
    vec4 pos = u_mvp * vec4(in_vert, 1.0);
    pos.z -= u_bias;
    gl_Position = pos;
}

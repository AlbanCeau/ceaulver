#version 330

in vec3 in_vert;
uniform mat4 u_mvp;

void main() {
    gl_Position = u_mvp * vec4(in_vert, 1.0);
}

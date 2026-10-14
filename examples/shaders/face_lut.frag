#version 330

// Colour from a LUT indexed by the face's world-space orientation (lat-long), nothing else.
flat in vec3 v_nrm;
flat in vec3 v_tint;
uniform sampler2D u_lut;
out vec4 f_color;

const float PI = 3.14159265358979;

void main() {
    vec3 n = normalize(v_nrm);
    float u = atan(n.z, n.x) / (2.0 * PI) + 0.5;
    float v = acos(clamp(n.y, -1.0, 1.0)) / PI;
    f_color = vec4(texture(u_lut, vec2(u, v)).rgb * v_tint, 1.0);   // RGB LUT (grey LUTs are uploaded as RGB) x tint
}

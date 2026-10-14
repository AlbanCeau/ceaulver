#version 330

flat in vec3 v_nrm;
flat in vec3 v_tint;
in vec4 v_cur;
in vec4 v_prev;
uniform sampler2D u_lut;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec2 f_vel;     // screen motion over the shutter, in UV units

const float PI = 3.14159265358979;

void main() {
    vec3 n = normalize(v_nrm);
    float u = atan(n.z, n.x) / (2.0 * PI) + 0.5;
    float v = acos(clamp(n.y, -1.0, 1.0)) / PI;
    f_color = vec4(texture(u_lut, vec2(u, v)).rgb * v_tint, 1.0);   // alpha 1 = geometry here
    f_vel = (v_cur.xy / v_cur.w - v_prev.xy / v_prev.w) * 0.5;
}

#version 330

// Velocity-buffer motion blur. Blur direction = the LONGEST motion in a small
// neighbourhood (a simplified neighbour-max), so moving objects smear across their own
// silhouettes onto the background, not only inside themselves. Colour is then averaged
// along that motion, centred on the pixel (the shutter spans both sides of the frame).
in vec2 v_uv;
uniform sampler2D u_color;
uniform sampler2D u_vel;
uniform vec2 u_texel;         // 1 / framebuffer size
uniform float u_max_uv;       // clamp on blur length, UV units
uniform float u_dilate_px;    // neighbourhood radius for the max motion, pixels
uniform vec3 u_bg;
uniform int u_samples;
out vec4 f_color;

vec3 col(vec2 uv) {
    vec4 c = texture(u_color, uv);
    return c.a > 0.5 ? c.rgb : u_bg;       // cleared to 0: alpha 0 = background
}

void main() {
    vec2 v = texture(u_vel, v_uv).xy;
    float best = dot(v, v);
    for (int i = 0; i < 8; i++) {
        float a = 6.2831853 * float(i) / 8.0;
        vec2 o = vec2(cos(a), sin(a)) * u_dilate_px * u_texel;
        vec2 vn = texture(u_vel, v_uv + o).xy;
        float d = dot(vn, vn);
        if (d > best) { best = d; v = vn; }
    }
    float len = sqrt(best);
    if (len > u_max_uv) v *= u_max_uv / len;
    if (len * min(1.0 / u_texel.x, 1.0 / u_texel.y) < 0.5) {   // under half a pixel: no blur
        f_color = vec4(col(v_uv), 1.0);
        return;
    }
    vec3 acc = vec3(0.0);
    for (int k = 0; k < u_samples; k++) {
        float t = float(k) / float(u_samples - 1) - 0.5;
        acc += col(v_uv + v * t);
    }
    f_color = vec4(acc / float(u_samples), 1.0);
}

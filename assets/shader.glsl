
---VERTEX SHADER-------------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

attribute vec3  v_pos;
attribute vec3  v_normal;
attribute vec2  v_tex_coord;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;

varying vec2 tex_coord0;
varying vec4 normal_vec;
varying vec4 vertex_pos;

void main (void) {
    vec4 pos = modelview_mat * vec4(v_pos, 1.0);
    vertex_pos = pos;
    normal_vec = vec4(v_normal, 0.0);
    gl_Position = projection_mat * pos;
    tex_coord0 = v_tex_coord;
}

---FRAGMENT SHADER-----------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

varying vec2 tex_coord0;
varying vec4 normal_vec;
varying vec4 vertex_pos;

uniform sampler2D texture_id;
uniform float brightness;
uniform float contrast;
// uniform float camera_distance;
uniform vec3 center_point;
// uniform float segment_fog_factor;
// uniform float dist_to_center;
uniform float material_density;
uniform float fog_density;
uniform float fog_radius;
// uniform vec3 fog_color;
uniform mat4 normal_mat;

void main (void){
    // vec4 back_tex_color = texture2D(background_texture_id, tex_coord0).rgba;
    // vec3 camera_pos = vec3(0.0, 0.0, 0.0);
    // float camera_distance = length(camera_pos - vertex_pos.xyz);
    // float fog_factor = 1.0 / exp(camera_distance * fog_density);
    // float fog_factor = exp(-pow(camera_distance * fog_density, 2.0));
    // float center_point_distance = length(center_point - vertex_pos.xyz) - camera_distance;
    // float fog_factor = 1.0 / exp(center_point_distance * fog_density);
    // float fog_factor = exp(-pow(center_point_distance * fog_density, 2.0));
    // float fog_factor = 1.0;
    // float fog_factor = exp(-pow(center_point_distance * fog_density, 2.0));
    // float fog_factor = 1.0 / exp(center_point_distance * fog_density * (1.0 - segment_fog_factor) * (1.0 - segment_fog_factor));    
    // fog_factor = fog_factor * fog_factor * fog_factor;
    // fog_factor = clamp(fog_factor, 0.0, 1.0);
    // float fog_factor = clamp(segment_fog_factor, 0.0, 1.0);
    // float fog_factor = exp(-pow(camera_distance * fog_density, 2.0));
    // float fog_factor = exp(-pow(dist_to_center * fog_density, 2.0));
    // float fog_factor = 0.5 - dist_to_center / 24.0;
    // fog_factor = clamp(fog_factor, 0.0, 1.0);
    // float distance = ( 1.0 - segment_fog_factor ) * ( 1.0 - segment_fog_factor ) * ( 1.0 - segment_fog_factor ) * 24.0;
    // float fog_factor = exp(-pow(distance * fog_density, 2.0));
    // fog_factor = clamp(fog_factor, 0.0, 1.0);
    // vec3 color_with_fog = mix(fog_color, tex_color.rgb, fog_factor);
    // vec3 color_with_fog = tex_color.rgb;
    // vec3 result_color = mix(fog_color, tex_color.rgb, fog_factor);
    // result_color = (result_color - 0.5) * contrast + 0.5 + brightness;
    // float result_alpha = tex_color.a;

    vec4 tex_color = texture2D(texture_id, tex_coord0).rgba;
    float center_point_distance = length(center_point - vertex_pos.xyz);
    center_point_distance = clamp(center_point_distance - fog_radius, 0.0, center_point_distance);
    float fog_factor = exp(-pow(center_point_distance * fog_density, 2.0));
    fog_factor = clamp(fog_factor + material_density, 0.0, 1.0);

    vec3 result_color = (tex_color.rgb - 0.5) * contrast + 0.5 + brightness;
    float result_alpha = tex_color.a * fog_factor;

    gl_FragColor = vec4(result_color, result_alpha);
}

#version 430

layout(local_size_x = 64) in;

//layout(local_size_x = 32, local_size_y = 32) in;

// Declare input buffers (SSBOs)
layout(std430, binding = 0) buffer PointsA {
    vec4 points_a[];
};

layout(std430, binding = 1) buffer PointsB {
    vec4 points_b[];
};

layout(std430, binding = 2) buffer ScanLines {
    uvec4 scan_lines[];
};

layout(std430, binding = 3) buffer Correspondences {
    vec4 corr[];
};

layout(std430, binding = 4) buffer Hessians {
    float Hs[][8][8];
};

layout(std430, binding = 5) buffer Bside {
    float Bs[][8];
};


uniform vec4 origin;
uniform vec4 debug_info;
uniform uvec4 lens_data;


void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    uint idx = id.x;
    uint idy = id.y;


    if (idx >= lens_data.y) {
        return;
    }

    //code//
}

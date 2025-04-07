#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 6) out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

const float triangleSize = 0.01; // Adjust for visibility
out float fragDepth;

in vec3 geomColor[];
out vec3 fragColor;

void main() {
    vec3 right = vec3(view[0][0], view[1][0], view[2][0]) * triangleSize;
    vec3 up = vec3(view[0][1], view[1][1], view[2][1]) * triangleSize;

    // Center of the triangle
    vec3 center = gl_in[0].gl_Position.xyz;

    // Billboard triangle vertices
    vec3 v1 = center - right - up;  // Bottom-left
    vec3 v2 = center + right - up;  // Bottom-right
    vec3 v3 = center - right + up;          // Top-left

    // Transform offsets using model, view, projection
    gl_Position = projection * view * model * vec4(v1, 1.0);
    fragDepth = -(view * model * vec4(v1, 1.0)).z;
    fragColor = geomColor[0];
    EmitVertex();

    gl_Position = projection * view * model * vec4(v2, 1.0);
    fragDepth = -(view * model * vec4(v2, 1.0)).z;
    fragColor = geomColor[0];
    EmitVertex();

    gl_Position = projection * view * model * vec4(v3, 1.0);
    fragDepth = -(view * model * vec4(v3, 1.0)).z;
    fragColor = geomColor[0];
    EmitVertex();

    EndPrimitive();



    // Second triangle (top-left, top-right, bottom-right)
    v1 = center - right + up;  // Top-left
    v2 = center + right + up;  // Top-right
    v3 = center + right - up;  // Bottom-right

    gl_Position = projection * view * model * vec4(v1, 1.0);
    fragDepth = -(view * model * vec4(v1, 1.0)).z;
    fragColor = geomColor[0];
    EmitVertex();

    gl_Position = projection * view * model * vec4(v2, 1.0);
    fragDepth = -(view * model * vec4(v2, 1.0)).z;
    fragColor = geomColor[0];
    EmitVertex();

    gl_Position = projection * view * model * vec4(v3, 1.0);
    fragDepth = -(view * model * vec4(v3, 1.0)).z;
    fragColor = geomColor[0];
    EmitVertex();

    EndPrimitive();
}
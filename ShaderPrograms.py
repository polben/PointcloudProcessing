TEST_VERTEX_SHADER = """
#version 330 core

layout (location = 0) in vec3 inPosition; // Center of the point
layout (location = 1) in vec3 inColor;

out vec3 geomColor;

void main() {
    // Pass through position to the geometry shader
    gl_Position = vec4(inPosition, 1.0);
    geomColor = inColor;
}
"""

CUBE_GEOMETRY_SHADER = """

#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 36) out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

in vec3 geomColor[];

out float fragDepth;
out vec3 fragColor;

void emitQuad(vec3 v1, vec3 v2, vec3 v3, vec3 v4, float colorBrightness) {
    gl_Position = projection * view * model * vec4(v1, 1.0);
    fragDepth = -(view * model * vec4(v1, 1.0)).z;
    fragColor = geomColor[0] * colorBrightness;
    EmitVertex();
    
    gl_Position = projection * view * model * vec4(v2, 1.0);
    fragDepth = -(view * model * vec4(v2, 1.0)).z;
    fragColor = geomColor[0] * colorBrightness;
    EmitVertex();

    gl_Position = projection * view * model * vec4(v3, 1.0);
    fragDepth = -(view * model * vec4(v3, 1.0)).z;
    fragColor = geomColor[0] * colorBrightness;
    EmitVertex();
    
    
    ///t2
    
    gl_Position = projection * view * model * vec4(v1, 1.0);
    fragDepth = -(view * model * vec4(v3, 1.0)).z;
    fragColor = geomColor[0] * colorBrightness;
    EmitVertex();

    gl_Position = projection * view * model * vec4(v3, 1.0);
    fragDepth = -(view * model * vec4(v2, 1.0)).z;
    fragColor = geomColor[0] * colorBrightness;
    EmitVertex();

    gl_Position = projection * view * model * vec4(v4, 1.0);
    fragDepth = -(view * model * vec4(v4, 1.0)).z;
    fragColor = geomColor[0] * colorBrightness;
    EmitVertex();
    
    EndPrimitive();
    
}

uniform float voxelSize;

void main() {
    
    vec3 center = gl_in[0].gl_Position.xyz;

    float s = voxelSize / 2.0;

    //vertices
    vec3 v0 = center + vec3(-s, -s, -s);
    vec3 v1 = center + vec3( s, -s, -s);
    vec3 v2 = center + vec3( s,  s, -s);
    vec3 v3 = center + vec3(-s,  s, -s);
    vec3 v4 = center + vec3(-s, -s,  s);
    vec3 v5 = center + vec3( s, -s,  s);
    vec3 v6 = center + vec3( s,  s,  s);
    vec3 v7 = center + vec3(-s,  s,  s);
    
    float bright = 1.0;
    float dark = 0.1;
    float bright1 = 0.2;
    float bright2 = 0.2 * 2;
    float bright3 = 0.2 * 3;
    float bright4 = 0.2 * 4;
    
    emitQuad(v0, v1, v2, v3, bright4); // Front  
    emitQuad(v5, v4, v7, v6, bright1); // Back  
    emitQuad(v4, v0, v3, v7, bright3); // Left  
    emitQuad(v1, v5, v6, v2, bright2); // Right  
    emitQuad(v3, v2, v6, v7, bright);  // Top  
    emitQuad(v0, v1, v5, v4, dark);    // Bottom


}

"""

TEST_GEOMETRY_SHADER = """
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
"""

TEST_FRAGMENT_SHADER = """
#version 330 core

out vec4 FragColor;
in float fragDepth;

float viewRange = 1000.0;

in vec3 fragColor;

void main() {

    float depthNormalized = 1.0 - clamp(fragDepth / viewRange, 0.0, 1.0);
    FragColor = vec4(fragColor * depthNormalized, 1.0);
}
"""

CUBE_FRAGMENT_SHADER = """
#version 330 core

out vec4 FragColor;
in float fragDepth;
in vec3 fragColor;

float viewRange = 1000.0;

void main() {

    float depthNormalized = 1.0 - clamp(fragDepth / viewRange, 0.0, 1.0);
    FragColor = vec4(fragColor * depthNormalized, 1.0);
}
"""


DEFAULT_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inColor;


uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;

void main() {
    gl_Position = projection * view * model * vec4(inPosition, 1.0);
    fragColor = inColor;
}
"""

# Fragment Shader
DEFAULT_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

in vec3 fragColor;

void main() {
    FragColor = vec4(fragColor, 1.0); // White color
}
"""
#version 330 core

layout (location = 0) in vec3 inPosition; // Center of the point
layout (location = 1) in vec3 inColor;

out vec3 geomColor;

void main() {
    // Pass through position to the geometry shader
    gl_Position = vec4(inPosition, 1.0);
    geomColor = inColor;
}
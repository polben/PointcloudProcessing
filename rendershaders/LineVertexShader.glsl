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
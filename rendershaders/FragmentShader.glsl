#version 330 core

out vec4 FragColor;
in float fragDepth;

float viewRange = 1000.0;

in vec3 fragColor;

void main() {

    float depthNormalized = 1.0 - clamp(fragDepth / viewRange, 0.0, 1.0);
    FragColor = vec4(fragColor * depthNormalized, 1.0);
}
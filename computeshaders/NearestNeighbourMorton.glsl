#version 430

layout(local_size_x = 64) in;

// Declare input buffers (SSBOs)
layout(std430, binding = 0) buffer MortonBufferA {
    uint a_sorted_mortons[];  // Sorted Morton codes
};

layout(std430, binding = 1) buffer PointBufferA {
    vec4 a_sorted_points[];  // Sorted points (3D coordinates)
};

// Declare input buffers (SSBOs)
layout(std430, binding = 2) buffer MortonBufferB {
    uint b_mortons[];  // Sorted Morton codes
};

layout(std430, binding = 3) buffer PointBufferB {
    vec4 b_points[];  // Sorted points (3D coordinates)
};

layout(std430, binding = 4) buffer Correspondences {
    uint corr[];  // Sorted Morton codes
};


uint naiveClosest(vec4 point){
    uint closestIndex = 0;
    float minDist = 9999999.0;

    for(uint i = 0; i < a_sorted_points.length(); i++){
        float dist = dot(a_sorted_points[i] - point, a_sorted_points[i] - point);
        if(dist < minDist){
            closestIndex = i;
            minDist = dist;
        }
    }


    return closestIndex;
}

uint mortonClosest(vec4 point_b, uint morton_b){
    int a_len = a_sorted_mortons.length() - 1;
    int low = 0;
    int high = a_len;

    int close_index = -1;
    while (low <= high){
        int mid = (low + (high - low) / 2);

        if (a_sorted_mortons[mid] == morton_b){
            close_index = mid;
            break;
        }

        if (a_sorted_mortons[mid] < morton_b){
            low = mid + 1;
        }else{
            high = mid - 1;
        }
    }

    if (close_index == -1){
        close_index = low;
    }

    close_index = clamp(close_index, 0, a_len);

    float PERCENT_OF_POINTS_TO_CHECK = 0.01 / 100.0;
    int numPointsCheck = int(a_len * PERCENT_OF_POINTS_TO_CHECK);

    int start = max(0, close_index - numPointsCheck);
    int end = min(a_len, close_index + numPointsCheck);

    float mindist = 1e20;
    for (int i = start; i <= end; i++){
        vec4 reference_point = a_sorted_points[i];

        float ddist = dot(vec3(point_b) - vec3(reference_point), vec3(point_b) - vec3(reference_point));
        if (ddist < mindist){
            close_index = i;
            mindist = ddist;
        }
    }

    return uint(close_index);
}


void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= b_mortons.length()) {
        return;
    }

    uint code_b = b_mortons[id];
    vec4 point_b = b_points[id];


    // corr[id] = naiveClosest(point_b);
    corr[id] = mortonClosest(point_b, code_b);
}

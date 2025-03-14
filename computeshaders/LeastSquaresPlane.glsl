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


layout(std430, binding = 4) buffer Hessians {
    float Hs[][8][8];
};

layout(std430, binding = 5) buffer Bside {
    float Bs[][8];
};

layout(std430, binding = 6) buffer NormalsA {
    vec4 normals_a[];
};

uniform vec4 origin;
uniform vec4 debug_info;
uniform uvec4 lens_data;


vec4 projectSpherePoint(vec4 point){
    float lidar_tall = 0.254;
    vec4 lidarOffset = vec4(0, lidar_tall / 2.0, 0, 0);
    vec4 projvec = point - origin - lidarOffset;
    return projvec / length(projvec);
}

float mcos(int index, vec4 refvec){
    float lidar_tall = 0.254;
    vec4 lidarOffset = vec4(0, lidar_tall / 2.0, 0, 0);

    vec4 circVec = points_a[index] - origin - lidarOffset;
    //circVec[1] = 0;
    circVec = circVec / length(circVec);
    return dot(circVec, refvec);
}

int indmod(int index, int clen, int cbegin) {
    return (index + clen) % clen;
}

int dir(int index, int circlen, int begin, vec4 refVec) {
    int p1 = begin + indmod(index - 1, circlen, begin);
    int p2 = begin + indmod(index + 1, circlen, begin);


    float cos1 = mcos(p1, refVec) + 1;
    float cos2 = mcos(p2, refVec) + 1;

    if (cos1 > cos2) {
        return -1;
    } else {
        return 1;
    }
}
int binaryScanCircleSearch(int begin, int end, vec4 refp) {
    vec4 refVec = projectSpherePoint(refp);

    int circlen = end - begin + 1;
    int index = (begin + end) / 2;
    int half_c = circlen / 2;

    int runs = int(ceil(log2(circlen)));

    for (int i = 0; i < runs; i++){
        int d = dir(index, circlen, begin, refVec);
        index = indmod(index + int(half_c * d), circlen, begin);
        half_c = int(ceil(half_c / 2.0));
    }

    return begin + index;


}


uint naiveClosest(vec3 point){
    uint closestIndex = 0;
    float minDist = 1.23e20;

    for(uint i = 0; i < lens_data.x; i++){
        float dist = dot(vec3(points_a[i]) - point, vec3(points_a[i]) - point);
        if(dist < minDist){
            closestIndex = i;
            minDist = dist;
        }
    }


    return closestIndex;
}

float distsq(vec4 a, vec4 b){
    vec4 diff = a - b;
    return dot(diff, diff);
}



int findClosestScanLine(vec4 refpoint){
    float height = projectSpherePoint(refpoint).y;

    int closest = 0;

    float minhdiff = 1.23e20;

    for (int i = 0; i < int(lens_data.z); i++){
        int begin = int(scan_lines[i][0]);

        vec4 ap = projectSpherePoint(points_a[begin]);

        float htdiff = abs(ap.y - height);
        if (htdiff < minhdiff){
            minhdiff = htdiff;
            closest = i;
        }
    }

    return closest;
}

vec2 binsearchAndCheck(int scanLine, vec4 refPoint){
    int begin = int(scan_lines[scanLine][0]);
    int end = int(scan_lines[scanLine][1]);
    int closest = binaryScanCircleSearch(begin, end, refPoint);


    int check = 20;
    int circlen = end - begin + 1;
    int index = closest - begin;

    float mindist = 1.23e20;

    for (int i = -check; i < check; i++) {
        int test = begin + indmod(index + i, circlen, begin);

        float d = distsq(refPoint, points_a[test]);
        if (d < mindist) {
            mindist = d;
            closest = test;
        }
    }

    return vec2(float(closest), float(mindist));
}







void getJ(vec4 point, vec4 normal, out float j[6]) {
    float nx = normal.x;
    float ny = normal.y;
    float nz = normal.z;
    float x = point.x;
    float y = point.y;
    float z = point.z;

    // Point-to-Plane Jacobian (1x6 row vector)
    j[0] = nx;                    // Translation X
    j[1] = ny;                    // Translation Y
    j[2] = nz;                    // Translation Z
    j[3] = ny * z - nz * y;        // Rotation X
    j[4] = nz * x - nx * z;        // Rotation Y
    j[5] = nx * y - ny * x;        // Rotation Z
}

void getH(in float[6] J_in, uint idx, float valid) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float sum_val = J_in[i] * J_in[j]; // J^T J for a 1x6 Jacobian
            Hs[idx][i][j] = sum_val * valid;
        }
    }
}

void getB(in float[6] J_i, in float e_i, uint idx, float valid) {
    for (int i = 0; i < 6; i++) {
        float sum_val = J_i[i] * e_i; // J^T * e
        Bs[idx][i] = sum_val * valid;
    }

}



void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    uint idx = id.x;
    uint idy = id.y;

    if (idx >= lens_data.y) {
        return;
    }

    /*Hs[idx][0][0] = -12.3;
    Hs[idx][0][1] = float(lens_data.x);
    Hs[idx][0][2] = float(lens_data.y);
    Hs[idx][0][3] = float(lens_data.z);
    Hs[idx][0][4] = float(idx);
    Hs[idx][7][7] = -1.0;

    Hs[idx][1][0] = normals_a[idx][0];
    Hs[idx][1][1] = normals_a[idx][1];
    Hs[idx][1][2] = normals_a[idx][2];

    Bs[idx][0] = -999.3;
    Bs[idx][1] = float(idx);
    Bs[idx][2] = float(lens_data.x);
    Bs[idx][3] = float(lens_data.y);
    Bs[idx][4] = float(lens_data.z);*/

    float minDist = 1.23e20;
    int minind = 0;
    vec4 refPoint = points_b[idx];
    int min_scan = 0;


    int closest_scan = findClosestScanLine(refPoint);
    int scans_to_check = 10;


    for (int i = max(0, closest_scan - scans_to_check); i <= min(int(lens_data.z) - 1, closest_scan + scans_to_check); i++) {
        vec2 res = binsearchAndCheck(i, refPoint);
        if (res.y < minDist) {
            minDist = res.y;
            minind = int(res.x);
            min_scan = i;
        }
    }


    vec4 normal = normals_a[minind];

    float outlierError = 10;

    vec3 p1 = vec3(points_a[minind]);
    vec3 p2 = vec3(refPoint);

    float valid = 0.0;
    if (minDist < outlierError) {
        valid = 1.0;
    }else {
        valid = 0.0;
    }

    float err = dot(vec3(normal), (p1 - p2));

    float[6] j;
    getJ(points_a[minind], normal, j);

    getH(j, idx, valid);

    getB(j, err, idx, valid);



}

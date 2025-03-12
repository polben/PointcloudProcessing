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
    float Hs[][6][6];
};

layout(std430, binding = 5) buffer Bside {
    float Bs[][6];
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

    for(uint i = 0; i < points_a.length(); i++){
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

    int low = 0;
    int high = scan_lines.length() - 1;
    int closest = 0;

    float minhdiff = 1.23e20;

    for (int i = 0; i <= high; i++){
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







void getJ(vec3 point, out float j[3][6]) {
    float x = point[0];
    float y = point[1];
    float z = point[2];

    // Filling the matrix based on the provided formula
    j[0][0] = 1.0; j[0][1] = 0.0; j[0][2] = 0.0;    j[0][3] = 0.0; j[0][4] = -z; j[0][5] = y;
    j[1][0] = 0.0; j[1][1] = 1.0; j[1][2] = 0.0;    j[1][3] = z;  j[1][4] = 0.0; j[1][5] = -x;
    j[2][0] = 0.0; j[2][1] = 0.0; j[2][2] = 1.0;    j[2][3] = -y; j[2][4] = x;  j[2][5] = 0.0;

}

void getH(in float[3][6] J_in, out float h[6][6], uint idx, float valid) {

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float sum_val = 0.0;
            for (int k = 0; k < 3; k++) {
                sum_val += J_in[k][i] * J_in[k][j];
            }
            //h[i][j] = sum_val;
            Hs[idx][i][j] = sum_val * valid;
        }
    }
}

void getB(in float[3][6] J_i, in float[3] e_i, out float b[6], uint idx, float valid) {
    for (int i = 0; i < 6; i++) {
        float sum_val = 0.0;
        for (int k = 0; k < 3; k++) {
            sum_val += J_i[k][i] * e_i[k];
        }
        //b[i] = sum_val;
        Bs[idx][i] = sum_val * valid;
    }
}

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    uint idx = id.x;
    uint idy = id.y;


    if (idx >= lens_data[1]) { // length of points_b
        return;
    }

    float minDist = 1.23e20;
    int minind = 0;
    vec4 refPoint = points_b[idx];

    if (debug_info[0] < 1.0) {

        int closest_scan = findClosestScanLine(refPoint);
        int scans_to_check = 10;


        for (int i = max(0, closest_scan - scans_to_check); i <= min(scan_lines.length() - 1, closest_scan + scans_to_check); i++) {
            vec2 res = binsearchAndCheck(i, refPoint);
            if (res.y < minDist) {
                minDist = res.y;
                minind = int(res.x);
            }
        }
    }else{
        minDist = 0.0;
        minind = int(idx);
    }



    float outlierError = 0.1;

    vec3 p1 = vec3(points_a[minind]);
    vec3 p2 = vec3(refPoint);

    float valid = 0.0;
    if (minDist < outlierError) {
        valid = 1.0;
    }else {
        valid = 0.0;
    }

    float[3] err;
    err[0] = p1.x - p2.x;
    err[1] = p1.y - p2.y;
    err[2] = p1.z - p2.z;

    float[3][6] j;
    getJ(p2, j);

    float[6][6] h;
    getH(j, h, idx, valid);

    float[6] b;
    getB(j, err, b, idx, valid);

    /*
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                Hs[idx][i][j] = h[i][j];
            }
        }

        // Set the vector b in the Bside SSBO at the calculated index (one element at a time)
        for (int i = 0; i < 6; i++) {
            Bs[idx][i] = b[i];
        }
    }else{
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                Hs[idx][i][j] = 0;
            }
        }

        // Set the vector b in the Bside SSBO at the calculated index (one element at a time)
        for (int i = 0; i < 6; i++) {
            Bs[idx][i] = 0;
        }
    }*/

}

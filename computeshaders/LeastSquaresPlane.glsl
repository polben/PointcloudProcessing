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

    vec4 refPoint = points_b[idx];
    float minDist = 1.23e20;

    int minind = findClosestPoint(refPoint, idx);
    minDist = distsq(refPoint, points_a[minind]);


    /*int minind = 0;
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
    }*/


    vec4 normal = normals_a[minind];

    float outlierError = 0.1;

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

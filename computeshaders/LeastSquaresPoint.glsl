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


    if (idx >= lens_data.y) { // length of points_b
        return;
    }

    float minDist = 1.23e20;
    int minind = 0;
    vec4 refPoint = points_b[idx];

    if (debug_info[0] < 1.0) {

        minind = findClosestPoint(refPoint, idx);
        minDist = distsq(refPoint, points_a[minind]);
        /*int closest_scan = findClosestScanLine(refPoint);
        int scans_to_check = 10;


        for (int i = max(0, closest_scan - scans_to_check); i <= min(int(lens_data.z) - 1, closest_scan + scans_to_check); i++) {
            vec2 res = binsearchAndCheck(i, refPoint);
            if (res.y < minDist) {
                minDist = res.y;
                minind = int(res.x);
            }
        }*/

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
}

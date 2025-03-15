void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    uint idx = id.x;
    uint idy = id.y;

    if (idx >= points_b.length()) {
        return;
    }



    vec4 refPoint = points_b[idx];

    /*int closest_scan = findClosestScanLine(refPoint);
    int scans_to_check = 2;

    float minDist = 1.23e20;
    int minind = 0;
    for (int i = max(0, closest_scan - scans_to_check); i <= min(scan_lines.length() - 1, closest_scan + scans_to_check); i++){
        vec2 res = binsearchAndCheck(i, refPoint);
        if(res.y < minDist){
            minDist = res.y;
            minind = int(res.x);
        }
    }*/

    int minind = findClosestPoint(refPoint);
    float minDist = distsq(refPoint, points_a[minind]);




    corr[idx][0] = float(minind);
    corr[idx][1] = minDist;
}

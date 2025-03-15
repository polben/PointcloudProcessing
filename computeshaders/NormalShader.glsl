int getScanLineOfPoint(uint point_id){
    for(int i = 0; i < int(lens_data.z); i++){
        uint begin = scan_lines[i][0];
        uint end = scan_lines[i][1];
        if (begin <= point_id && point_id <= end){
            return i;
        }

    }
}

int closestDistinctOnLine(uint pindex, uint scanline){
    int begin = int(scan_lines[scanline][0]);
    int end = int(scan_lines[scanline][1]);
    vec4 refp = points_a[pindex];

    int closest = binaryScanCircleSearch(begin, end, refp);
    float mindist = 1.23e20;
    int check = 5;
    int index = closest - begin;
    int circlen = int(end - begin) + 1;

    for (int i = -check; i < check; i++){
        int test = begin + indmod(index + i, circlen, begin);
        float d = distsq(refp, points_a[test]);

        if (d < mindist && test != int(pindex)){
            mindist = d;
            closest = test;
        }
    }

    return closest;
}

int getClosestAboveOrBelow(uint id, int scan){
    int scancount = int(lens_data.z);
    vec4 refp = points_a[id];

    if (scan == scancount - 1){
        return closestDistinctOnLine(id, scan - 1);
    }

    if (scan == 0){
        return closestDistinctOnLine(id, scan + 1);
    }

    int minabove = closestDistinctOnLine(id, scan - 1);
    int minbelow = closestDistinctOnLine(id, scan + 1);

    vec4 min_point_above = points_a[minabove];
    vec4 min_point_below = points_a[minbelow];

    if (distsq(refp, min_point_below) < distsq(refp, min_point_above)){
        return minbelow;
    }else{
        return minabove;
    }

}

float normalTowardsOrigin(uint id, vec3 normal){
    vec3 to_origin_n = vec3(origin - points_a[id]);
    to_origin_n = to_origin_n / length(to_origin_n);

    if (dot(normal, to_origin_n) > dot(-normal, to_origin_n)){
        return 1.0;
    }else{
        return -1.0;
    }
}

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    uint idx = id.x;
    uint idy = id.y;


    if (idx >= lens_data.x) { // length of points_b // dont use .a, it will break the comparison, its the _ALPHA_ of the rgb component :))))))
        return;
    }

    vec4 refp = points_a[idx];
    int minscan = getScanLineOfPoint(idx);
    int closest_on_line = closestDistinctOnLine(idx, minscan);
    int closest_next = getClosestAboveOrBelow(idx, minscan);

    vec4 point_line = points_a[closest_on_line];
    vec4 point_next = points_a[closest_next];
    vec4 point = refp;

    vec3 v1 = vec3(point_line - point);
    vec3 v2 = vec3(point_next - point);
    vec3 cr = cross(v1, v2);

    vec3 normal = cr / length(cr);
    normal *= normalTowardsOrigin(idx, normal);

    normals_a[idx] = vec4(normal, 0);

}

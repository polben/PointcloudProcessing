#version 430

layout(local_size_x = 64) in;


layout(std430, binding = 0) buffer PointsA {
    vec4 points_a[];
};



layout(std430, binding = 2) buffer ScanLines {
    uvec4 scan_lines[];
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

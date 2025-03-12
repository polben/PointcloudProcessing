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

uniform vec4 origin;

float mcos(int index, vec4 refvec){
    vec4 circVec = points_a[index] - origin;
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
    vec4 refVec = refp - origin;
    //refVec[1] = 0;
    refVec = refVec / length(refVec); //normalize

    int circlen = end - begin + 1;
    int index = (begin + end) / 2;
    int half_c = circlen / 2;

    while(half_c >= 1){
        int d = dir(index, circlen, begin, refVec);
        index = indmod(index + int(half_c * d), circlen, begin);
        half_c = half_c / 2;
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

ivec3 binaryScanLineSearch(vec4 refpoint){
    vec4 refvec = refpoint - origin;

    float height = (refvec / length(refvec)).y;

    int low = 0;
    int high = scan_lines.length() - 1; // -1 for last element, -1 for last begin
    int closest = 0;

    float minhdiff = 1.23e20;
    ivec3 minhind = ivec3(0, 0, 0); // 0: begin, 1: end, 2: index

    //for (int i = 0; i <= high; i++){
    while(low < high){
        int mid = (low + high) / 2;
        //int scanindex = i * 2;
        int scanindex = mid;

        int begin = int(scan_lines[scanindex][0]);
        int end = int(scan_lines[scanindex][1]);

        closest = binaryScanCircleSearch(begin, end, refpoint);
        vec4 ap = points_a[closest] - origin;

        ap = ap / length(ap);

        float htdiff = abs(ap.y - height);
        if (htdiff < minhdiff){
            minhdiff = htdiff;
            minhind = ivec3(begin, end, closest);
        }
        /*float ddiff = distsq(points_a[closest], refpoint);
        if (ddiff < minhdiff){
            minhdiff = ddiff;
            minhind = ivec3(begin, end, closest);
        }*/


        if (ap[1] > height){
            low = mid + 1;
        }else{
            high = mid;
        }
    }

    return minhind;
}

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    uint idx = id.x;
    uint idy = id.y;

    if (idx >= points_b.length()) {
        return;
    }



    vec4 refPoint = points_b[idx];

    /*uint closest = naiveClosest(vec3(refPoint));
    corr[idx][0] = float(closest);
    corr[idx][1] = length(refPoint - points_a[closest]);
    return;*/


    ivec3 closest = binaryScanLineSearch(refPoint);
    //corr[idx][0] = float(closest.z);
    //return;


    int check = 10;
    int begin = closest.x;
    int end = closest.y;
    int circlen = end - begin + 1;
    int index = closest.z - begin;

    float mindist = 1.23e20;

    for (int i = -check; i < check; i++) {
        int test = begin + indmod(index + i, circlen, begin);

        float d = distsq(refPoint, points_a[test]);
        if (d < mindist) {
            mindist = d;
            closest.z = test;
        }
    }




    corr[idx][0] = float(closest.z);
    corr[idx][1] = length(refPoint - points_a[closest.z]);



}

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
    /*while(low < high){
        int mid = (low + high) / 2;
        int scanindex = mid;

        int begin = int(scan_lines[scanindex][0]);

        vec4 ap = projectSpherePoint(points_a[begin]);

        float htdiff = abs(ap.y - height);
        if (htdiff < minhdiff){
            minhdiff = htdiff;
            closest = scanindex;
        }

        if (ap[1] > height){
            low = mid + 1;
        }else{
            high = mid;
        }
    }*/

    return closest;
}

vec2 binsearchAndCheck(int scanLine, vec4 refPoint){
    int begin = int(scan_lines[scanLine][0]);
    int end = int(scan_lines[scanLine][1]);
    int closest = binaryScanCircleSearch(begin, end, refPoint);


    int check = 100;
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

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    uint idx = id.x;
    uint idy = id.y;

    if (idx >= points_b.length()) {
        return;
    }



    vec4 refPoint = points_b[idx];

    int closest_scan = findClosestScanLine(refPoint);
    int scans_to_check = 2;

    float minDist = 1.23e20;
    int minind = 0;
    for (int i = max(0, closest_scan - scans_to_check); i <= min(scan_lines.length() - 1, closest_scan + scans_to_check); i++){
        vec2 res = binsearchAndCheck(i, refPoint);
        if(res.y < minDist){
            minDist = res.y;
            minind = int(res.x);
        }
    }




    corr[idx][0] = float(minind);
    corr[idx][1] = minDist;



}

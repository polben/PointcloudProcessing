#version 430

layout(local_size_x = 64) in;

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
    float Hs[][8][8];
};

layout(std430, binding = 5) buffer Bside {
    float Bs[][8];
};

layout(std430, binding = 6) buffer NormalsA {
    vec4 normals_a[];
};

layout(std430, binding = 7) buffer NormalsB {
    vec4 normals_b[];
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


    int check = 10;
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


int fullBinary(vec4 refpoint){
    int scansLen_m1 = int(lens_data.z) - 1;
    int upperScan = 0;
    int lowerScan = scansLen_m1;

    float targetHeight = projectSpherePoint(refpoint).y;
    int prevmid = -1;
    int mid = 0;

    int closest_scan = 0;
    while(prevmid != mid){
        prevmid = mid;
        mid = (lowerScan + upperScan) / 2;
        int begin = int(scan_lines[mid][0]);


        int global_idx_online_a = begin; //binaryScanCircleSearch(begin, end, refpoint);
        float height = projectSpherePoint(points_a[global_idx_online_a]).y;
        if (height < targetHeight){
            lowerScan = mid;
        }else{
            upperScan = mid;
        }
        closest_scan = mid;
    }


    int scans_to_check = 1;

    float minDist = 1.23e20;
    int minind = 0;
    for (int i = max(0, closest_scan - scans_to_check); i <= min(scansLen_m1, closest_scan + scans_to_check); i++){
        vec2 res = binsearchAndCheck(i, refpoint);
        if(res.y < minDist){
            minDist = res.y;
            minind = int(res.x);
        }
    }
    return minind;
}

int findClosestPoint(vec4 refpoint, uint id){
    float mind = 1.23e20;
    int glob_closest = 0;

    float last_scan_line = corr[id][3];


    int from = 0;
    int to = int(lens_data.z);

    int scan_check_width = 2;
    if (last_scan_line != -1.0){
        from = max(0, int(last_scan_line) - scan_check_width);
        to = min(int(lens_data.z), int(last_scan_line) + scan_check_width);
    }

    for (int i = from; i < to; i++) {

        vec2 closest = binsearchAndCheck(i, refpoint);

        if (closest.y < mind){
            mind = closest.y;
            glob_closest = int(closest.x);
            corr[id][3] = float(i);
        }
    }

    return glob_closest;
}




void main(){}
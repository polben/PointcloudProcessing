#version 430

layout(local_size_x = 64) in;

layout(std430, binding = 6) buffer VoxelIndex {
    ivec4 voxel_index[];
};

layout(std430, binding = 7) buffer VoxelData {
    int voxel_data[][1024];
};

layout(std430, binding = 10) buffer RenderStage {
    int voxel_stage_data[][1024];
};

layout(std430, binding = 11) buffer CounterBuffer {
    int counter_buffer[][16];
};

layout(std430, binding = 12) buffer VoxelStatBuffer {
    vec4 voxel_stat[];
};

// using voxel_stat buffer for probabilities and other stagind data:
// prob, prev_points, ready to stage, staged
// prob, prev_points, (will be staged but few points, ready to stage), staged
// (will be staged, but few points) > will be staged: 0.2, ready to stage: 0.8, staged: 0.0 / 1.0
// will be staged: has reaced probability treshold once


// first element of debug buffer, last entry ([3]) number of elements buffert to staging area
// when treshold is crossed, copy data into stage buffer from voxel data
// concurrent conter: counter buffer


float BayesFilter(float prob_prev, float alpha, float beta) {
    return (alpha * prob_prev) / (alpha * prob_prev + beta * (1.0 - prob_prev));
}

bool tryStageVoxel(int idx, int point_count, int max_vox_points, int max_stage){
    int current = atomicAdd(counter_buffer[0][0], 1); // t1: gets 0, t2: gets 1 > at this point, counter is 2
    if (current < max_stage){ // t1: 0 < 1 >> stage

        // voxel_stage_data[current][0] = 6969;
        voxel_stage_data[current][0] = point_count;

        for (int i = 1; i < min(point_count + 1, max_vox_points); i++){
            voxel_stage_data[current][i] = voxel_data[idx][i];
        }

        return true;


    }else{ // t2: 1 !< 1: revert counter from 2 to 1, returns false
        atomicAdd(counter_buffer[0][0], -1);
        return false;
    }

}




int binSearch(int axis, int value, bool findFirst, int from, int to){
    int l = from;
    int h = to;

    int first_occ = -1;

    while(l <= h){
        int mid = int((l + h) / 2.0);
        if (voxel_index[mid][axis] == value){
            first_occ = mid;
            if (findFirst){
                h = mid - 1;
            }else{
                l = mid + 1;
            }

        }else if (voxel_index[mid][axis] < value){
            l = mid + 1;
        }else{
            h = mid - 1;
        }
    }

    return first_occ;
}

int findVoxelId(ivec3 voxel_coord){
    if (int(counter_buffer[0][3]) == 0){ // stored_voxel_num
        return -1;
    }

    int x_val = voxel_coord[0];
    int y_val = voxel_coord[1];
    int z_val = voxel_coord[2];

    int axis_x = 0;
    int axis_y = 1;
    int axis_z = 2;

    int first_occ_x = binSearch(axis_x, x_val, true, 0, int(counter_buffer[0][3]) - 1);
    if (first_occ_x == -1){
        return -1;
    }

    int last_occ_x = binSearch(axis_x, x_val, false, 0, int(counter_buffer[0][3]) - 1);

    int first_occ_y = binSearch(axis_y, y_val, true, first_occ_x, last_occ_x);
    if (first_occ_y == -1){
        return -1;
    }
    int last_occ_y = binSearch(axis_y, y_val, false, first_occ_x, last_occ_x);


    int first_occ_z = binSearch(axis_z, z_val, true, first_occ_y, last_occ_y);
    if (first_occ_z == -1){
        return -1;
    }

    return first_occ_z;

}

int getPointCount(int idx){
    float max_prob_prev_points = voxel_stat[idx][1];
    float max_prob = max_prob_prev_points - float(int(max_prob_prev_points));
    return int(max_prob_prev_points);
}

bool groundRule1(bool vfound[6], bool vstatic[6]){
    return
    ( vstatic[2] && vstatic[3] && vstatic[4] && vstatic[5] ) ||
    ( !vstatic[2] && vstatic[3] && vstatic[4] && vstatic[5] ) ||
    ( vstatic[2] && !vstatic[3] && vstatic[4] && vstatic[5] ) ||
    ( vstatic[2] && vstatic[3] && !vstatic[4] && vstatic[5] ) ||
    ( vstatic[2] && vstatic[3] && vstatic[4] && !vstatic[5] );
}

bool groundRule2(bool vfound[6], bool vstatic[6], int indices[6]){
    int hfound = 0;
    for (int i = 2; i < 6; i++){
        if (vstatic[i]){
            hfound++;
        }
    }
    if (hfound >= 2){
        if(!vfound[1]){
            return true;
        }
    }

    return false;
}

bool recoverVoxel(ivec4 currentVoxelId, int idx){
    ivec3 up = ivec3(0, 1, 0);
    ivec3 down = ivec3(0, -1 ,0);
    ivec3 side1 = ivec3(1, 0, 0);
    ivec3 side2 = ivec3(-1, 0, 0);
    ivec3 side3 = ivec3(0, 0, 1);
    ivec3 side4 = ivec3(0, 0, -1);
    ivec3 sides[] = {up, down, side1, side2, side3, side4};
    bool vfound[6];
    bool vstatic[6];
    int indices[6];

    ivec3 this_voxel_id = ivec3(currentVoxelId);

    for (int i = 0; i < 6; i++){
        int found = findVoxelId(this_voxel_id + sides[i]);
        indices[i] = found;
        if (found != -1){
            vfound[i] = true;
            int voxel_stat_id = voxel_index[found][3];
            float render_status = voxel_stat[voxel_stat_id][3];
            float stage_status = voxel_stat[voxel_stat_id][2];

            if (render_status > 0.5 || stage_status > 0.1){ // voxel next to this: either rendered, or ready to stage
                vstatic[i] = true;
            }else{
                vstatic[i] = false;
            }
        }else{
            vfound[i] = false;
        }
    }

    if(groundRule1(vfound, vstatic) || groundRule2(vfound, vstatic, indices)){
        return true;
    }


    return false;
}



void main() {

    uvec2 id = gl_GlobalInvocationID.xy;
    int idx = int(id.x);
    uint idy = id.y;

    int stored_voxel_num = int(counter_buffer[0][3]);
    int max_vox_points = int(counter_buffer[0][2]);
    int max_staging = int(counter_buffer[0][1]);

    float alpha = 0.55;
    float beta = 0.45;

    if (idx >= stored_voxel_num) {
        return;
    }

    ivec4 voxelId = voxel_index[idx];
    idx = voxel_index[idx][3];



    float is_staged = voxel_stat[idx][3];

    if (is_staged > 0.5){ // if a voxel was staged in a previous run (rendered) just return
        return;
    }

    int point_count = voxel_data[idx][0];
    //int prev_count = int(voxel_stat[idx][1]);
    float max_prob_prev_points = voxel_stat[idx][1];
    float max_prob = max_prob_prev_points - float(int(max_prob_prev_points));
    int prev_count = int(max_prob_prev_points);


    // if a voxel has ready to stage status, but was not staged because there was not enough place the prev iteration, stage it
    // but if there is not enough place again, try the next iteration
    float stage_status = voxel_stat[idx][2];

    if (counter_buffer[0][4] == 1){ // if full stage request, mark all voxels ready to stage
        if (stage_status > 0.1) {
            voxel_stat[idx][2] = 0.8;
        }else{
            if (counter_buffer[0][5] != 1 && recoverVoxel(voxelId, idx)){
                voxel_stat[idx][2] = 0.8;
            }
        }
    }

    /*voxel_stage_data[10 + idx][0] = idx + 1;
    if (idx == 1){
        voxel_stage_data[5][0] = 69;
        int a = 0;
        for (int i = 0; i < 1000000; i++){
            if (a % 123 == 0){
                a = i;
            }
            a += 1;
            voxel_stage_data[5][1] = a;
            voxel_stage_data[6][idx] = idx;
        }
    }*/

    if (stage_status > 0.5){ // (0.8 for ready to stage, 0.2 will be staged
        if(tryStageVoxel(idx, point_count, max_vox_points, max_staging)){
            voxel_stat[idx][3] = 1.0; // voxel was succesfully staged
        }
        return;
    }








    // from here, these are voxels that:
    // are not ready to stage: either still accumulating points, or are not (yet) statc: stage status is 0.0 or 0.2
    float old_prob = voxel_stat[idx][0];

    float new_prob = 0.0;
    if (point_count == prev_count) {
        new_prob = BayesFilter(old_prob, 1.0 - alpha, 1.0 - beta);
    }else{
        new_prob = BayesFilter(old_prob, alpha, beta);
    }
    if (new_prob > max_prob){
        max_prob = new_prob;
    }
    voxel_stat[idx][0] = new_prob;

    if (max_prob > 0.999){
        max_prob = 0.999;
    }
    if(max_prob < 0.001){
        max_prob = 0.001;
    }
    voxel_stat[idx][1] = float(point_count) + max_prob;

    // staging status here is either: 0.0 or 0.2
    // probability is either > 0.9 or below
    if (new_prob > 0.9 || stage_status > 0.1 || counter_buffer[0][5] == 1){ // pass for voxels that has just reached 0.9 static probability, or will be rendered, OR not filtering for outliers
        if (stage_status < 0.1){ // this voxel will be rendered, set if newly 0.9 and not yet marked as to be rendered
            voxel_stat[idx][2] = 0.2;
        }



        // ready to stage when: max points is reached (discard: or no longer seeing it > prob < 0.5)
        if (point_count + 1 >= max_vox_points){
            voxel_stat[idx][2] = 0.8;
        }

    }
}
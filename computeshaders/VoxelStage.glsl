#version 430

layout(local_size_x = 64) in;

layout(std430, binding = 9) buffer VoxelData {
    int voxel_data[][1024];
};




layout(std430, binding = 12) buffer RenderStage {
    int voxel_stage_data[][1024];
};

layout(std430, binding = 13) buffer CounterBuffer {
    int counter_buffer[][16];
};

layout(std430, binding = 14) buffer VoxelStatBuffer {
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

bool tryStageVoxel(int idx, int point_count, int max_stage){
    int current = atomicAdd(counter_buffer[0][0], 1); // t1: gets 0, t2: gets 1 > at this point, counter is 2
    if (current < max_stage){ // t1: 0 < 1 >> stage

        // voxel_stage_data[current][0] = 6969;
        voxel_stage_data[current][0] = point_count;

        for (int i = 1; i < point_count; i++){
            voxel_stage_data[current][i] = voxel_data[idx][i];
        }

        return true;


    }else{ // t2: 1 !< 1: revert counter from 2 to 1, returns false
        atomicAdd(counter_buffer[0][0], -1);
        return false;
    }

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

    /*float p = voxel_stat[idx][0];
    if (p < 0.5){
        p = 2.0;
    }else{
        p *= 2.0;
    }
    voxel_stat[idx][0] = p;
    return;*/


    float is_staged = voxel_stat[idx][3];

    if (is_staged > 0.5){ // if a voxel was staged in a previous run (rendered) just return
        return;
    }

    if (is_staged < -0.5){ // voxel had high probability of being moving
        return;
    }

    int point_count = voxel_data[idx][0];
    int prev_count = int(voxel_stat[idx][1]);



    // if a voxel has ready to stage status, but was not staged because there was not enough place the prev iteration, stage it
    // but if there is not enough place again, try the next iteration
    float stage_status = voxel_stat[idx][2];

    if (counter_buffer[0][4] == 1){ // if full stage request, mark all voxels ready to stage
        if (stage_status > 0.1) {
            voxel_stat[idx][2] = 0.8;
        }
    }

    if (stage_status > 0.5){ // (0.8 for ready to stage, 0.2 will be staged
        if(tryStageVoxel(idx, min(point_count, max_vox_points), max_staging)){
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
    voxel_stat[idx][0] = new_prob;
    voxel_stat[idx][1] = float(point_count);

    // staging status here is either: 0.0 or 0.2
    // probability is either > 0.9 or below
    if (new_prob > 0.95 || stage_status > 0.1){ // pass for voxels that has just reached 0.9 static probability, or will be rendered
        if (stage_status < 0.1){ // this voxel will be rendered, set if newly 0.9 and not yet marked as to be rendered
            voxel_stat[idx][2] = 0.2;
        }



        // ready to stage when: max points is reached or no longer seeing it
        if (new_prob < 0.5 || point_count + 1 >= max_vox_points){
            voxel_stat[idx][2] = 0.8;
        }

    }

    if (new_prob < 0.2){
        voxel_stat[idx][3] = -1.0;
    }
}
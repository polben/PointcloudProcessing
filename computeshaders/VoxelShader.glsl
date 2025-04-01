#version 430

layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer PointsA {
    vec4 points[];
};

layout(std430, binding = 8) buffer VoxelIndex {
    ivec4 voxel_index[];
};

layout(std430, binding = 9) buffer VoxelData {
    int voxel_data[][1024];
};

layout(std430, binding = 10) buffer UnknownPointInds {
    int unknown_point_inds[];
};


layout(std430, binding = 11) buffer DebugBuffer {
    vec4 debug[];
};


uniform uvec4 voxel_lens_data;
uniform float voxel_size;


ivec3 getVoxelCoord(vec4 point){ // vec3-vec4?
    vec4 vcoord = ceil(point / voxel_size);
    return ivec3(int(vcoord[0]), int(vcoord[1]), int(vcoord[2]));
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
    if (voxel_lens_data[1] == 0){
        return -1;
    }

    int x_val = voxel_coord[0];
    int y_val = voxel_coord[1];
    int z_val = voxel_coord[2];

    int axis_x = 0;
    int axis_y = 1;
    int axis_z = 2;

    int first_occ_x = binSearch(axis_x, x_val, true, 0, int(voxel_lens_data[1]) - 1);
    if (first_occ_x == -1){
        return -1;
    }

    int last_occ_x = binSearch(axis_x, x_val, false, 0, int(voxel_lens_data[1]) - 1);

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

void main(){

    uvec2 id = gl_GlobalInvocationID.xy;
    uint idx = id.x;
    uint idy = id.y;

    if (idx >= voxel_lens_data[0]) {
        return;
    }

    debug[idx][3] = float(idx);

    int current_max_points = int(voxel_lens_data[0]);
    int current_stored_voxels = int(voxel_lens_data[1]);
    int batch_begin_index = int(voxel_lens_data[2]);

    vec4 point = points[idx];
    ivec3 voxelCoord = getVoxelCoord(point);
    debug[idx][0] = float(voxelCoord[0]);
    debug[idx][1] = float(voxelCoord[1]);
    debug[idx][2] = float(voxelCoord[2]);



    int voxel_index_entry = findVoxelId(voxelCoord);
    debug[idx][3] = float(voxel_index_entry);

    if (voxel_index_entry == -1){
        int unknown_count = atomicAdd(unknown_point_inds[0], 1);
        unknown_point_inds[1 + unknown_count] = int(voxel_lens_data[2]) + int(idx);
    }else{
        int voxel_data_index = voxel_index[voxel_index_entry][3];
        int stored_points_num = atomicAdd(voxel_data[voxel_data_index][0], 1);
        if (stored_points_num < int(voxel_lens_data[3]) - 1) {
            voxel_data[voxel_data_index][1 + stored_points_num] = int(voxel_lens_data[2]) + int(idx);
        }
    }

}
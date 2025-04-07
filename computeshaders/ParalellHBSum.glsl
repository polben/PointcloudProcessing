


// float Hs[][8][8]; bound from base!!


layout(std430, binding = 15) buffer ConcurrentHB1 {
    float HB1[][8][8];
};

layout(std430, binding = 16) buffer ConcurrentHB2 {
    float HB2[][8][8];
};

layout(std430, binding = 11) buffer DebugBuffer {
    vec4 debug[];
};

uniform uvec4 properties;

shared float sdata[64][8][8];

bool writeTargetIs2(){
    return int(properties[1]) == 0;
}

void copyToShared(int globalId, int localId){
    int point_count = int(properties[0]);

    if (globalId < point_count) {

        if (writeTargetIs2()) {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    //sdata[localId][i][j] = HB1[globalId][i][j];
                    float value = Hs[globalId][i][j];
                    if(!isnan(value)) {
                        sdata[localId][i][j] = Hs[globalId][i][j];
                    }else{
                        sdata[localId][i][j] = 0.0;
                    }
                }
            }
        } else {
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    float value = HB2[globalId][i][j];
                    if (!isnan(value)) {
                        sdata[localId][i][j] = value;
                    }else{
                        sdata[localId][i][j] = 0.0;
                    }
                }
            }
        }
    }else{
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                sdata[localId][i][j] = 0.0;
            }
        }
    }
}

void addToLocal(int local_to, int local_from){
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            sdata[local_to][i][j] += sdata[local_from][i][j];
        }
    }
}

void copyShared0ToGlobal(int global_index){
    if (writeTargetIs2()) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                HB2[global_index][i][j] = sdata[0][i][j];
            }
        }
    }else{
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                //HB1[global_index][i][j] = sdata[0][i][j];
                Hs[global_index][i][j] = sdata[0][i][j];
            }
        }
    }
}


void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int lid = int(gl_LocalInvocationID.x);
    int block_id = int(gl_WorkGroupID.x);

    int point_count = int(properties[0]);
    int block_size = int(properties[3]);

    copyToShared(gid, lid);
    barrier();

    // # 1
    /*for (int s = 1; s < block_size; s *= 2){
        if(lid % (2 * s) == 0){
            addToLocal(lid, lid + s);
        }
        barrier();
    }*/

    // # 2
    /*for (int s = 1; s < block_size; s *= 2){
        int index = 2 * s * lid;
        if (index < block_size){
            addToLocal(index, index + s);
        }
        barrier();
    }*/

    // # 3
    for (int s = block_size / 2; s > 0; s>>=1){
        if (lid < s){
            addToLocal(lid, lid + s);
        }

        barrier();
    }



    if (lid == 0){
        copyShared0ToGlobal(block_id);
    }

}
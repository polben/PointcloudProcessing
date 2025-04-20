


// float Hs[][8][8]; bound from base!!


layout(std430, binding = 13) buffer ConcurrentHB2 {
    float HB2[][8][8];
};

uniform uvec4 properties;

shared float sdata[64][8][8];

void copyToShared(int globalId, int localId, bool targetIs2){
    int point_count = int(properties[0]);

    if (globalId < point_count) {

        if (targetIs2) {
            for (int i = 0; i < 6; i++) {
                vec4 a = vec4(Hs[globalId][i][0], Hs[globalId][i][1],
                              Hs[globalId][i][2], Hs[globalId][i][3]);
                vec4 b = vec4(Hs[globalId][i][4], Hs[globalId][i][5],
                              Hs[globalId][i][6], Hs[globalId][i][7]);

                sdata[localId][i][0] = a.x; sdata[localId][i][1] = a.y;
                sdata[localId][i][2] = a.z; sdata[localId][i][3] = a.w;
                sdata[localId][i][4] = b.x; sdata[localId][i][5] = b.y;
                sdata[localId][i][6] = b.z; sdata[localId][i][7] = b.w;
                /*for (int j = 0; j < 8; j++) {
                    //sdata[localId][i][j] = HB1[globalId][i][j];

                    sdata[localId][i][j] = Hs[globalId][i][j];

                }*/
            }
        } else {
            for (int i = 0; i < 6; i++) {
                vec4 a = vec4(HB2[globalId][i][0], HB2[globalId][i][1],
                              HB2[globalId][i][2], HB2[globalId][i][3]);
                vec4 b = vec4(HB2[globalId][i][4], HB2[globalId][i][5],
                              HB2[globalId][i][6], HB2[globalId][i][7]);

                sdata[localId][i][0] = a.x; sdata[localId][i][1] = a.y;
                sdata[localId][i][2] = a.z; sdata[localId][i][3] = a.w;
                sdata[localId][i][4] = b.x; sdata[localId][i][5] = b.y;
                sdata[localId][i][6] = b.z; sdata[localId][i][7] = b.w;
                /*for (int j = 0; j < 8; j++) {

                    sdata[localId][i][j] = HB2[globalId][i][j];

                }*/
            }
        }
    }else{
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 7; j++) {
                sdata[localId][i][j] = 0.0;
            }
        }
    }
}

void addToLocal(int local_to, int local_from){
    for (int i = 0; i < 6; i++) {
        vec4 t_a = vec4(sdata[local_to][i][0], sdata[local_to][i][1],
                        sdata[local_to][i][2], sdata[local_to][i][3]);
        vec4 t_b = vec4(sdata[local_to][i][4], sdata[local_to][i][5],
                        sdata[local_to][i][6], sdata[local_to][i][7]);
        vec4 s_a = vec4(sdata[local_from][i][0], sdata[local_from][i][1],
                        sdata[local_from][i][2], sdata[local_from][i][3]);
        vec4 s_b = vec4(sdata[local_from][i][4], sdata[local_from][i][5],
                        sdata[local_from][i][6], sdata[local_from][i][7]);
        // Add them together.
        t_a += s_a;
        t_b += s_b;
        // Write back.
        sdata[local_to][i][0] = t_a.x; sdata[local_to][i][1] = t_a.y;
        sdata[local_to][i][2] = t_a.z; sdata[local_to][i][3] = t_a.w;
        sdata[local_to][i][4] = t_b.x; sdata[local_to][i][5] = t_b.y;
        sdata[local_to][i][6] = t_b.z; sdata[local_to][i][7] = t_b.w;
        /*for (int j = 0; j < 8; j++) {
            sdata[local_to][i][j] += sdata[local_from][i][j];
        }*/
    }
}

void copyShared0ToGlobal(int global_index, bool targetIs2){

    if (targetIs2) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 7; j++) {
                vec4 a = vec4(sdata[0][i][0], sdata[0][i][1],
                      sdata[0][i][2], sdata[0][i][3]);
                vec4 b = vec4(sdata[0][i][4], sdata[0][i][5],
                      sdata[0][i][6], sdata[0][i][7]);

                HB2[global_index][i][0] = a.x; HB2[global_index][i][1] = a.y;
                HB2[global_index][i][2] = a.z; HB2[global_index][i][3] = a.w;
                HB2[global_index][i][4] = b.x; HB2[global_index][i][5] = b.y;
                HB2[global_index][i][6] = b.z; HB2[global_index][i][7] = b.w;
                //HB2[global_index][i][j] = sdata[0][i][j];
            }
        }
    }else{
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 7; j++) {
                vec4 a = vec4(sdata[0][i][0], sdata[0][i][1],
                      sdata[0][i][2], sdata[0][i][3]);
                vec4 b = vec4(sdata[0][i][4], sdata[0][i][5],
                      sdata[0][i][6], sdata[0][i][7]);

                //HB1[global_index][i][j] = sdata[0][i][j];
                //Hs[global_index][i][j] = sdata[0][i][j];
                Hs[global_index][i][0] = a.x; Hs[global_index][i][1] = a.y;
                Hs[global_index][i][2] = a.z; Hs[global_index][i][3] = a.w;
                Hs[global_index][i][4] = b.x; Hs[global_index][i][5] = b.y;
                Hs[global_index][i][6] = b.z; Hs[global_index][i][7] = b.w;

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

    bool targetIs2 = int(properties[1]) == 0;

    copyToShared(gid, lid, targetIs2);
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
        copyShared0ToGlobal(block_id, targetIs2);
    }

}
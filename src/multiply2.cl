const int TS = 32;

kernel void multiply(
    const int N,
    const global float* A,
    const global float* B,
    global float* C
) {
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col;

    local float ASub[TS][TS];
    local float BSub[TS][TS];

    float value = 0.;
    const int numTiles = N / TS;
    for (int tile = 0; tile < numTiles; tile++) {
        const int tiledRow = TS * tile + row;
        const int tiledCol = TS * tile + col;

        ASub[col][row] = A[tiledCol * N + globalRow];
        BSub[col][row] = B[globalCol * N + tiledRow];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            value += ASub[k][row] * BSub[col][k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[globalCol * N + globalRow] = value;
}

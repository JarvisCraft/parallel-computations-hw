const int TS = 32;
const int WPT = 8;
const int RTS = TS / WPT;

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

    float values[WPT];
    for (int index = 0; index < WPT; index++) {
        values[index] = 0.;
    }

    const int numTiles = N / TS;
    for (int tile = 0; tile < numTiles; tile++) {
        for (int index = 0; index < WPT; index++) {
            const int tiledRow = TS * tile + row;
            const int tiledCol = TS * tile + col;
            const int offset = index * RTS;
            const int subCol = col + offset;

            ASub[subCol][row] = A[(tiledCol + offset) * N + globalRow];
            BSub[subCol][row] = B[(globalCol + offset) * N + tiledRow];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            for (int index = 0; index < WPT; index++) {
                values[index] += ASub[k][row] * BSub[col + index * RTS][k];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int index = 0; index < WPT; index++) {
        C[(globalCol + index * RTS) * N + globalRow] = values[index];
    }
}

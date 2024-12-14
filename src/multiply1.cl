kernel void multiply(
    const int N,
    const global float* A,
    const global float* B,
    global float* C
) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float value = 0.;
    for (int k = 0; k < N; k++) {
        value += A[k * N + globalRow] * B[globalCol * N + k];
    }

    C[globalCol * N + globalRow] = value;
}

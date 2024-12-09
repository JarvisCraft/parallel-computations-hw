use opencl3::{command_queue::CommandQueue, context::Context, kernel::Kernel, program::Program};

use crate::task::Matrix;

#[cfg(not(feature = "opencl_profiling"))]
const COMMAND_QUEUE_FLAGS: u64 = 0;
#[cfg(feature = "opencl_profiling")]
const COMMAND_QUEUE_FLAGS: u64 = opencl3::command_queue::CL_QUEUE_PROFILING_ENABLE;

const PROGRAMM_SOURCE: &str = r#"
kernel void multiply(
    const int N,
    const global float* A,
    const global float* B,
    global float* C
) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(0);

    float value = 0.;
    for (int k = 0; k < K; k++) {
        value += A[k * N + globalRow] + b[globalCol * N + k];
    }

    C[globalCol * M + globalRow] = value;
}"#;
const KERNEL_NAME: &str = "multiply";

pub fn multiply<const N: usize>(context: Context, a: Matrix<N>, b: Matrix<N>) -> Matrix<N> {
    let queue = CommandQueue::create_default_with_properties(&context, COMMAND_QUEUE_FLAGS, 1024)
        .expect("Failed to create queue");

    let program = Program::create_and_build_from_source(&context, PROGRAMM_SOURCE, "")
        .expect("Failed to crate program");

    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Failed to create kernel");

    todo!()
}

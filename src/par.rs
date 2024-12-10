use std::ptr;

use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    program::Program,
    types::{cl_float, cl_int, CL_BLOCKING, CL_NON_BLOCKING},
};

use crate::{task::Matrix, types::ZERO};

const COMMAND_QUEUE_FLAGS: u64 = opencl3::command_queue::CL_QUEUE_PROFILING_ENABLE;

const PROGRAM_SOURCE: &str = r#"
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
}"#;
const KERNEL_NAME: &str = "multiply";

pub fn multiply<const N: usize>(context: Context, a: Matrix<N>, b: Matrix<N>) -> Matrix<N> {
    let buffer_size = N * N;
    assert!(a.len() == buffer_size);
    assert!(b.len() == buffer_size);

    let n = cl_int::try_from(N).expect("N does not fit into cl_int");

    let queue = CommandQueue::create_default_with_properties(&context, COMMAND_QUEUE_FLAGS, 0)
        .expect("Failed to create queue");
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Failed to crate program");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Failed to create kernel");

    let mut a_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, buffer_size, ptr::null_mut())
    }
    .expect("Failed to create buffer for matrix A");
    let mut b_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, buffer_size, ptr::null_mut())
    }
    .expect("Failed to create buffer for matrix B");
    let c_buffer = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, buffer_size, ptr::null_mut())
    }
    .expect("Failed to create buffer for matrix B");

    let a = a.into_inner();
    let b = b.into_inner();

    let a_write_event =
        unsafe { queue.enqueue_write_buffer(&mut a_buffer, CL_BLOCKING, 0, a.as_ref(), &[]) }
            .expect("Failed to write A");
    let b_write_event =
        unsafe { queue.enqueue_write_buffer(&mut b_buffer, CL_BLOCKING, 0, b.as_ref(), &[]) }
            .expect("Failed to write B");

    let mut execute_kernel = ExecuteKernel::new(&kernel);
    let kernel_event = unsafe {
        execute_kernel
            .set_arg(&n)
            .set_arg(&a_buffer)
            .set_arg(&b_buffer)
            .set_arg(&c_buffer)
            .set_global_work_size(N)
            .set_global_work_size(N)
            .set_wait_event(&a_write_event)
            .set_wait_event(&b_write_event)
            .enqueue_nd_range(&queue)
    }
    .expect("Failed to create kernel event");

    let events = vec![kernel_event.get()];
    let mut result = vec![ZERO; buffer_size];
    let read_event =
        unsafe { queue.enqueue_read_buffer(&c_buffer, CL_NON_BLOCKING, 0, &mut result, &events) }
            .expect("Failed to wait for read event");

    read_event.wait().expect("Failed to wait for read event");
    let start_time = kernel_event
        .profiling_command_start()
        .expect("Failed to get start time");
    let end_time = kernel_event
        .profiling_command_end()
        .expect("Failed to get end time");
    println!("Took {} ns", end_time - start_time);

    result
        .into_boxed_slice()
        .try_into()
        .expect("Dimensions should match")
}

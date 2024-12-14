use std::ptr;

use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE},
    program::Program,
    types::{cl_command_queue_properties, cl_float, cl_int, CL_BLOCKING},
};

use crate::{
    task::{Matrix, Solution, Task},
    types::ZERO,
};

#[cfg(feature = "profiling")]
const COMMAND_QUEUE_FLAGS: cl_command_queue_properties =
    opencl3::command_queue::CL_QUEUE_PROFILING_ENABLE;
#[cfg(not(feature = "profiling"))]
const COMMAND_QUEUE_FLAGS: cl_command_queue_properties = 0;

const PROGRAM_SOURCE: &str = include_str!("multiply1.cl");
const KERNEL_NAME: &str = "multiply";
const LOCAL_WORK_SIZE: usize = 1;

pub struct Executor<const N: usize> {
    // context: Context,
    command_queue: CommandQueue,
    // program: Program,
    kernel: Kernel,
    n: cl_int,
    a_buffer: Buffer<cl_float>,
    b_buffer: Buffer<cl_float>,
    c_buffer: Buffer<cl_float>,
}
impl<const N: usize> Executor<N> {
    const BUFFER_SIZE: usize = N * N;

    pub fn new(context: Context) -> Self {
        let n = cl_int::try_from(N).expect("Failed to convert N to cl_int");

        let command_queue =
            CommandQueue::create_default_with_properties(&context, COMMAND_QUEUE_FLAGS, 0)
                .expect("Failed to create queue");
        let program = Program::create_and_build_from_source(
            &context,
            PROGRAM_SOURCE,
            opencl3::program::CL_STD_3_0,
        )
        .expect("Failed to create program");
        let kernel = Kernel::create(&program, KERNEL_NAME).expect("Failed to create kernel");

        let a_buffer: Buffer<f32> = unsafe {
            Buffer::<cl_float>::create(
                &context,
                CL_MEM_READ_ONLY,
                Self::BUFFER_SIZE,
                ptr::null_mut(),
            )
        }
        .expect("Failed to create buffer for matrix A");
        let b_buffer = unsafe {
            Buffer::<cl_float>::create(
                &context,
                CL_MEM_READ_ONLY,
                Self::BUFFER_SIZE,
                ptr::null_mut(),
            )
        }
        .expect("Failed to create buffer for matrix B");
        let c_buffer = unsafe {
            Buffer::<cl_float>::create(
                &context,
                CL_MEM_READ_WRITE,
                Self::BUFFER_SIZE,
                ptr::null_mut(),
            )
        }
        .expect("Failed to create buffer for matrix B");

        Self {
            // context,
            command_queue,
            // program,
            kernel,
            n,
            a_buffer,
            b_buffer,
            c_buffer,
        }
    }

    pub fn solve(&mut self, task: Task<N>) -> Solution<N> {
        let n = task.0.len();
        Solution(
            (0..n)
                .map(|index| {
                    self.multiply_all(task.0.iter().cycle().skip(index).take(n).cloned())
                        .expect("This is unrechable when `n` is zero")
                })
                .collect(),
        )
    }

    pub fn multiply_all(&mut self, matrices: impl Iterator<Item = Matrix<N>>) -> Option<Matrix<N>> {
        matrices.reduce(|l, r| self.multiply(&l, &r))
    }

    pub fn multiply(&mut self, a: &Matrix<N>, b: &Matrix<N>) -> Matrix<N> {
        let _a_write_event = unsafe {
            self.command_queue.enqueue_write_buffer(
                &mut self.a_buffer,
                CL_BLOCKING,
                0,
                a.as_slice(),
                &[],
            )
        }
        .expect("Failed to write A");
        let _b_write_event = unsafe {
            self.command_queue.enqueue_write_buffer(
                &mut self.b_buffer,
                CL_BLOCKING,
                0,
                b.as_slice(),
                &[],
            )
        }
        .expect("Failed to write B");

        let mut execute_kernel = ExecuteKernel::new(&self.kernel);
        let kernel_event = unsafe {
            execute_kernel
                .set_arg(&self.n)
                .set_arg(&self.a_buffer)
                .set_arg(&self.b_buffer)
                .set_arg(&self.c_buffer)
                .set_global_work_sizes(&[N / LOCAL_WORK_SIZE, N / LOCAL_WORK_SIZE])
                .enqueue_nd_range(&self.command_queue)
        }
        .expect("Failed to create kernel event");

        let events = vec![kernel_event.get()];
        let mut result = vec![ZERO; Self::BUFFER_SIZE];
        let _c_read_event = unsafe {
            self.command_queue.enqueue_read_buffer(
                &self.c_buffer,
                CL_BLOCKING,
                0,
                &mut result,
                &events,
            )
        }
        .expect("Failed to wait for read event");

        #[cfg(feature = "profiling")]
        {
            // Note: this will take time in execution time measurement.
            let start_time = kernel_event
                .profiling_command_start()
                .expect("Failed to get start time");
            let end_time = kernel_event
                .profiling_command_end()
                .expect("Failed to get end time");
            tracing::info!(
                "GPU task took {:?} ns",
                std::time::Duration::from_nanos(end_time - start_time)
            );
        }

        result
            .into_boxed_slice()
            .try_into()
            .expect("Dimensions should match")
    }
}

#[cfg(test)]
mod tests {
    use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};

    use super::*;

    fn device() -> Device {
        Device::new(
            *get_all_devices(CL_DEVICE_TYPE_GPU)
                .unwrap()
                .first()
                .unwrap(),
        )
    }

    #[test]
    fn simple_2x2() {
        let device = device();
        let context = Context::from_device(&device).unwrap();
        let mut executor = Executor::new(context);

        let a = Matrix::<2>::from_vec(vec![1., 3., 2., 4.]).unwrap();
        let b = Matrix::<2>::from_vec(vec![5., 7., 6., 8.]).unwrap();

        assert_eq!(
            executor.multiply(&a, &b),
            Matrix::from_vec(vec![19., 43., 22., 50.]).unwrap(),
        );
    }
}

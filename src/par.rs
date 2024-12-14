mod config;

use std::ptr;

use self::config::WorkSize;
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

const KERNEL_NAME: &str = "multiply";

pub struct Executor {
    work_size: WorkSize,
    // context: Context,
    command_queue: CommandQueue,
    // program: Program,
    kernel: Kernel,
    n: usize,
    n_int: cl_int,
    buffer_size: usize,
    a_buffer: Buffer<cl_float>,
    b_buffer: Buffer<cl_float>,
    c_buffer: Buffer<cl_float>,
}
impl Executor {
    pub fn new(n: usize, context: Context) -> Self {
        let buffer_size = n.checked_mul(n).expect("`n` is too big");
        let n_int = cl_int::try_from(n).expect("Failed to convert `n` to cl_int");
        let config = if n % 32 == 0 { config::V3 } else { config::V1 };

        let command_queue =
            CommandQueue::create_default_with_properties(&context, COMMAND_QUEUE_FLAGS, 0)
                .expect("Failed to create queue");
        let program = Program::create_and_build_from_source(
            &context,
            config.src,
            opencl3::program::CL_STD_3_0,
        )
        .expect("Failed to create program");
        let kernel = Kernel::create(&program, KERNEL_NAME).expect("Failed to create kernel");

        let a_buffer: Buffer<f32> = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, buffer_size, ptr::null_mut())
        }
        .expect("Failed to create buffer for matrix A");
        let b_buffer = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, buffer_size, ptr::null_mut())
        }
        .expect("Failed to create buffer for matrix B");
        let c_buffer = unsafe {
            Buffer::<cl_float>::create(&context, CL_MEM_READ_WRITE, buffer_size, ptr::null_mut())
        }
        .expect("Failed to create buffer for matrix B");

        Self {
            work_size: config.work_size,
            // context,
            command_queue,
            // program,
            kernel,
            n,
            n_int,
            buffer_size,
            a_buffer,
            b_buffer,
            c_buffer,
        }
    }

    pub fn solve(&mut self, task: &Task) -> Solution {
        assert!(task.n() == self.n, "Task simension shoud match this d");
        let n = task.matrices().len();

        Solution(
            (0..n)
                .map(|index| {
                    unsafe {
                        self.multiply_all_unchecked(
                            task.matrices().iter().cycle().skip(index).take(n).cloned(),
                        )
                    }
                    .expect("This is unrechable when `n` is zero")
                })
                .collect(),
        )
    }

    pub fn multiply_all(&mut self, matrices: impl Iterator<Item = Matrix>) -> Option<Matrix> {
        matrices.reduce(|l, r| self.multiply(&l, &r))
    }

    pub unsafe fn multiply_all_unchecked(
        &mut self,
        matrices: impl Iterator<Item = Matrix>,
    ) -> Option<Matrix> {
        matrices.reduce(|l, r| unsafe { self.multiply_unchecked(&l, &r) })
    }

    pub fn multiply(&mut self, a: &Matrix, b: &Matrix) -> Matrix {
        assert!(a.n() == b.n(), "Matrices should have the same dimensions");
        assert!(
            a.n() == self.n,
            "Matrices should be of dimmension {} but are of dimension {}",
            self.n,
            a.n()
        );

        unsafe { self.multiply_unchecked(a, b) }
    }
    pub unsafe fn multiply_unchecked(&mut self, a: &Matrix, b: &Matrix) -> Matrix {
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
                .set_arg(&self.n_int)
                .set_arg(&self.a_buffer)
                .set_arg(&self.b_buffer)
                .set_arg(&self.c_buffer)
                .set_global_work_sizes(&[self.n, self.n / self.work_size.per_thread]);
            if let Some(local) = self.work_size.local {
                execute_kernel
                    .set_local_work_sizes(&[local.get(), local.get() / self.work_size.per_thread]);
            }

            execute_kernel.enqueue_nd_range(&self.command_queue)
        }
        .expect("Failed to create kernel event");

        let events = vec![kernel_event.get()];
        let mut result = vec![ZERO; self.buffer_size];
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
        let mut executor = Executor::new(2, context);

        let a = Matrix::from_vec(vec![1., 3., 2., 4.]).unwrap();
        let b = Matrix::from_vec(vec![5., 7., 6., 8.]).unwrap();

        assert_eq!(
            executor.multiply(&a, &b),
            Matrix::from_vec(vec![19., 43., 22., 50.]).unwrap(),
        );
    }
}

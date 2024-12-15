pub mod config;

use std::iter;
use std::{num::NonZeroUsize, ptr};

use self::config::Config;
use self::config::WorkSize;
use opencl3::{
    command_queue::CommandQueue,
    context::Context,
    error_codes::ClError,
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

#[derive(thiserror::Error, Debug)]
pub enum NewExecutorError {
    #[error("dimension {0} is too big")]
    TooBig(usize),
    #[error("dimension {0} cannot be converted to OpenCL int")]
    InconvertibleN(usize),
    #[error("dimension should be multiple of {0} but is {1}")]
    UnsupportedSize(NonZeroUsize, usize),
    #[error("OpenCL failed: {0}")]
    ClError(#[from] ClError),
    #[error("failed to compile OpenCL program: {0}")]
    Compile(String),
}

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
    pub fn new(n: usize, context: &Context, config: Config) -> Result<Self, NewExecutorError> {
        let buffer_size = n.checked_mul(n).ok_or(NewExecutorError::TooBig(n))?;
        let n_int = cl_int::try_from(n).map_err(|_| NewExecutorError::InconvertibleN(n))?;

        let local_size = config
            .work_size
            .local
            .unwrap_or(const { NonZeroUsize::new(1).unwrap() });
        if n % local_size != 0 {
            return Err(NewExecutorError::UnsupportedSize(local_size, n));
        }

        let command_queue =
            CommandQueue::create_default_with_properties(context, COMMAND_QUEUE_FLAGS, 0)?;
        let program = Program::create_and_build_from_source(
            context,
            config.src,
            opencl3::program::CL_STD_3_0,
        )
        .map_err(NewExecutorError::Compile)?;
        let kernel = Kernel::create(&program, KERNEL_NAME)?;

        let a_buffer: Buffer<f32> = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_ONLY, buffer_size, ptr::null_mut())
        }?;
        let b_buffer = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_ONLY, buffer_size, ptr::null_mut())
        }?;
        let c_buffer = unsafe {
            Buffer::<cl_float>::create(context, CL_MEM_READ_WRITE, buffer_size, ptr::null_mut())
        }?;

        Ok(Self {
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
        })
    }

    pub fn solve(&mut self, task: &Task) -> Solution {
        assert!(
            task.n() == self.n,
            "Task dimension shoud match this solver's one"
        );
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

    pub fn solve_memoizing(&mut self, task: &Task) -> Solution {
        assert!(
            task.n() == self.n,
            "Task dimension shoud match this solver's one"
        );
        let n = task.matrices().len();
        let Some(first) = task.matrices().first() else {
            return Solution(vec![]);
        };

        let mut left_muls = Vec::with_capacity(n);
        left_muls.push(first.clone());
        for index in 0..(n - 1) {
            left_muls.push(unsafe {
                self.multiply_unchecked(&left_muls[index], &task.matrices()[index + 1])
            });
        }

        let last = task.matrices().last().unwrap();
        let mut right_muls = Vec::with_capacity(n - 1);
        right_muls.push(last.clone());
        for index in 0..n - 2 {
            right_muls.push(unsafe {
                self.multiply_unchecked(&task.matrices()[n - index - 2], &right_muls[index])
            })
        }

        Solution(
            // Start with `left_muls[n-1]` which is actually `A[0] * ... * A[n-1]`,
            //  then produce multiplications `right_muls[n-1 - (1..n)] * left_muls[1..n]`.
            iter::zip(
                left_muls.into_iter().cycle().skip(n - 1).take(n),
                [None].into_iter().chain(right_muls.iter().rev().map(Some)),
            )
            .map(|(left, right)| -> Matrix {
                if let Some(right) = right {
                    unsafe { self.multiply_unchecked(right, &left) }
                } else {
                    left
                }
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
        let mut executor = Executor::new(2, &context, config::V1).unwrap();

        let a = Matrix::from_vec(vec![1., 3., 2., 4.]).unwrap();
        let b = Matrix::from_vec(vec![5., 7., 6., 8.]).unwrap();

        assert_eq!(
            executor.multiply(&a, &b),
            Matrix::from_vec(vec![19., 43., 22., 50.]).unwrap(),
        );
    }
}

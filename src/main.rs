use std::time;

use opencl3::{
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
};
use task::{Matrix, Task};

mod cmd;
mod par;
mod seq;
mod task;
mod types;

fn main() {
    let device = Device::new(
        *get_all_devices(CL_DEVICE_TYPE_GPU)
            .expect("Failed to discover GPU devices")
            .first()
            .expect("There is no avalable GPU device"),
    );
    println!("Device: {device:?}");
    let context = Context::from_device(&device).expect("Failed to create context from device");

    let a = Matrix::<2>::from_vec(vec![1., 3., 2., 4.]).unwrap();
    let b = Matrix::<2>::from_vec(vec![5., 7., 6., 8.]).unwrap();

    let c_seq = seq::multiply(&a, &b);
    println!("C (sequential) = {c_seq:?}");
    let c_par = par::multiply(context, &a, &b);
    println!("C (parallel) = {c_par:?}");

    let task = Task(vec![a, b]);

    run(false, task);
}

fn run<const N: usize>(parallel: bool, task: Task<N>) {
    let begin = time::Instant::now();
    let result = if parallel {
        todo!("Implement parallel conputations");
    } else {
        seq::solve(task)
    };
    let end = time::Instant::now();

    println!("[{:?}] Result = {result:?}", end - begin);
}

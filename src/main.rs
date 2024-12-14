use std::time::{self, Duration};

use opencl3::{
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
};
use par::Executor;
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

    #[cfg(feature = "random")]
    let (a, b, c) = {
        const N: usize = 10;
        fn gen() -> Matrix<N> {
            use rand::prelude::*;
            let mut rng = rand::thread_rng();
            Matrix::from_vec((0..N * N).map(|_| rng.gen_range(-100. ..100.)).collect()).unwrap()
        }

        (gen(), gen(), gen())
    };
    #[cfg(not(feature = "random"))]
    let (a, b, c) = {
        let a = Matrix::<2>::from_vec(vec![1., 3., 2., 4.]).unwrap();
        let b = Matrix::<2>::from_vec(vec![5., 7., 6., 8.]).unwrap();
        let c = Matrix::<2>::from_vec(vec![9., 11., 10., 12.]).unwrap();
        (a, b, c)
    };
    let mut executor = Executor::new(context);

    let c_seq = seq::multiply(&a, &b);
    println!("C (sequential) = {c_seq:?}");
    let c_par = executor.multiply(&a, &b);
    println!("C (parallel) = {c_par:?}");

    let task = Task(vec![a, b, c]);
    let seq_time = run(Mode::Seq, task.clone(), &mut executor);
    let par_time = run(Mode::Par, task, &mut executor);

    println!("Sequential time: {seq_time:?}");
    println!("  Parallel time: {par_time:?}");
}

fn run<const N: usize>(mode: Mode, task: Task<N>, executor: &mut Executor<N>) -> Duration {
    let begin = time::Instant::now();
    let result = match mode {
        Mode::Seq => seq::solve(task),
        Mode::Par => executor.solve(task),
    };
    let end = time::Instant::now();

    println!("[{mode:?}] Result = {result:?}");

    end - begin
}

#[derive(Debug)]
enum Mode {
    Seq,
    Par,
}

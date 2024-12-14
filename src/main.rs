use std::time::{self, Duration};

use opencl3::{
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
};
use par::Executor;
use task::{Matrix, Task};
use tracing::{debug, info};

mod cmd;
mod par;
mod seq;
mod task;
mod types;

fn main() {
    tracing_subscriber::fmt::init();

    let device = pick_device().expect("There is no available GPU device");

    let context = Context::from_device(&device).expect("Failed to create context from device");

    #[cfg(feature = "random")]
    let (a, b, c) = {
        const N: usize = 1000;
        fn gen() -> Matrix<N> {
            use rand::prelude::*;
            let mut rng = rand::thread_rng();
            Matrix::from_vec(
                (0..N * N)
                    .map(|_| rng.gen_range(-100. ..100.))
                    .map(|v| 1. / v)
                    .collect(),
            )
            .unwrap()
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

    // TODO: move this to tests
    let c_seq = seq::multiply(&a, &b);
    debug!("C (sequential) = {c_seq:?}");
    let c_par = executor.multiply(&a, &b);
    debug!("C (parallel) = {c_par:?}");

    let task = Task(vec![a, b, c]);
    let seq_time = run(Mode::Seq, task.clone(), &mut executor);
    let par_time = run(Mode::Par, task, &mut executor);

    info!("Sequential time: {seq_time:?}");
    info!("  Parallel time: {par_time:?}");
}

fn pick_device() -> Option<Device> {
    let gpu_devices = get_all_devices(CL_DEVICE_TYPE_GPU).expect("Failed to discover GPU devices");
    info!("Avalable GPU devices: {gpu_devices:?}");

    get_all_devices(CL_DEVICE_TYPE_GPU)
        .expect("Failed to discover GPU devices")
        .first()
        .map(|id| Device::new(*id))
}

fn run<const N: usize>(mode: Mode, task: Task<N>, executor: &mut Executor<N>) -> Duration {
    let begin = time::Instant::now();
    let result = match mode {
        Mode::Seq => seq::solve(task),
        Mode::Par => executor.solve(task),
    };
    let end = time::Instant::now();

    debug!("[{mode:?}] Result = {result:?}");

    end - begin
}

#[derive(Debug)]
enum Mode {
    Seq,
    Par,
}

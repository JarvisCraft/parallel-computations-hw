use std::time::{self, Duration};

use opencl3::{
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
};
use par::Executor;
use task::{Matrix, Solution, Task};
use tracing::{debug, info};

mod cmd;
mod par;
mod seq;
mod task;
mod types;
mod util;

fn main() {
    tracing_subscriber::fmt::init();

    let device = pick_device().expect("There is no available GPU device");

    let context = Context::from_device(&device).expect("Failed to create context from device");

    #[cfg(feature = "random")]
    let (n, task) = {
        const N: usize = 512;
        fn gen() -> Matrix {
            use rand::prelude::*;
            let mut rng = rand::thread_rng();
            Matrix::from_vec(
                (0..N * N)
                    .map(|_| rng.gen_range(-1. ..0.))
                    .map(|v| -v)
                    .collect(),
            )
            .unwrap()
        }

        (N, Task::from_vec((0..8).map(|_| gen()).collect()).unwrap())
    };
    #[cfg(not(feature = "random"))]
    let (n, task) = {
        let a = Matrix::from_vec(vec![1., 3., 2., 4.]).unwrap();
        let b = Matrix::from_vec(vec![5., 7., 6., 8.]).unwrap();
        let c = Matrix::from_vec(vec![9., 11., 10., 12.]).unwrap();
        (
            2,
            Task::from_vec(vec![
                a.clone(),
                b.clone(),
                c.clone(),
                a.clone(),
                b.clone(),
                c.clone(),
                a,
                b,
                c,
            ])
            .unwrap(),
        )
    };
    let mut executor = Executor::new(n, context);

    let (seq_solution, seq_time) = run(Mode::Seq, &task, &mut executor);
    let (seq_par_solution, seq_par_time) = run(Mode::SeqPar, &task, &mut executor);
    let (par_solution, par_time) = run(Mode::Par, &task, &mut executor);

    info!(" Sequential time: {seq_time:?}");
    info!(
        " Sequential sol_: {} {:?}",
        seq_solution.0.len(),
        seq_solution.0[1][3]
    );

    info!("SeqParallel time: {seq_par_time:?}");
    info!(
        "SeqParallel sol_: {} {:?}",
        seq_par_solution.0.len(),
        seq_par_solution.0[1][3]
    );

    info!("   Parallel time: {par_time:?}");
    info!(
        "   Parallel sol_: {} {:?}",
        par_solution.0.len(),
        par_solution.0[1][3]
    );
}

fn pick_device() -> Option<Device> {
    let gpu_devices = get_all_devices(CL_DEVICE_TYPE_GPU).expect("Failed to discover GPU devices");
    info!("Avalable GPU devices: {gpu_devices:?}");

    get_all_devices(CL_DEVICE_TYPE_GPU)
        .expect("Failed to discover GPU devices")
        .first()
        .map(|id| Device::new(*id))
}

fn run(mode: Mode, task: &Task, executor: &mut Executor) -> (Solution, Duration) {
    info!("Running execution in {mode:?} mode");
    let begin = time::Instant::now();
    let solution = match mode {
        Mode::Seq => seq::solve(task),
        Mode::SeqPar => seq::solve_par(task),
        Mode::Par => executor.solve(task),
    };
    let end = time::Instant::now();

    debug!("[{mode:?}] Result = {solution:?}");
    info!("Completed execution in {mode:?} mode");

    (solution, end - begin)
}

#[derive(Debug)]
enum Mode {
    Seq,
    SeqPar,
    Par,
}

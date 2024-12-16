use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};

use clap::Parser;
use cmd::Cmd;
use comfy_table::Color;
use opencl3::{
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
};
use task::{Matrix, Solution, Task};
use tracing::{info, warn};

mod cmd;
mod par;
mod seq;
mod task;
mod types;
mod util;

fn main() {
    let Cmd {
        modes,
        checked_indices,
        matrices,
        dimension,
        sample,
    } = Cmd::parse();
    tracing_subscriber::fmt::init();

    let device = pick_device().expect("There is no available GPU device");
    let context = Context::from_device(&device).expect("Failed to create context from device");

    let task = if sample {
        let a = Matrix::from_vec(vec![1., 3., 2., 4.]).unwrap();
        let b = Matrix::from_vec(vec![5., 7., 6., 8.]).unwrap();
        let c = Matrix::from_vec(vec![9., 11., 10., 12.]).unwrap();

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
        .unwrap()
    } else {
        Task::from_vec(
            (0..matrices)
                .map(|_| {
                    use rand::prelude::*;
                    let mut rng = rand::thread_rng();
                    Matrix::from_vec(
                        (0..dimension * dimension)
                            .map(|_| rng.gen_range(-1. ..0.))
                            .map(|v| -v)
                            .collect(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    };

    let mut measurement = Measurement {
        context,
        modes: modes.into_iter().collect(),
    };
    measurement.run(&task, checked_indices);
}

fn pick_device() -> Option<Device> {
    let gpu_devices = get_all_devices(CL_DEVICE_TYPE_GPU).expect("Failed to discover GPU devices");
    info!("Available GPU devices: {gpu_devices:?}");

    get_all_devices(CL_DEVICE_TYPE_GPU)
        .expect("Failed to discover GPU devices")
        .first()
        .map(|id| Device::new(*id))
}

struct Measurement {
    context: Context,
    modes: BTreeSet<cmd::Mode>,
}
impl Measurement {
    pub fn run(&mut self, task: &Task, random_checks: usize) {
        let Self { context, modes } = &self;

        let mut verdicts = Vec::with_capacity(modes.len());
        for mode in modes {
            info!("[{mode:?}] Running execution");
            verdicts.push((
                *mode,
                match mode {
                    cmd::Mode::CpuSingleThreaded => Some(Self::run_cpu(false, task)),
                    cmd::Mode::CpuMultiThreaded => Some(Self::run_cpu(true, task)),
                    cmd::Mode::GpuNaive1 => Self::run_gpu(context, false, par::config::V1, task),
                    cmd::Mode::GpuNaive2 => Self::run_gpu(context, false, par::config::V2, task),
                    cmd::Mode::GpuNaive3 => Self::run_gpu(context, false, par::config::V3, task),
                    cmd::Mode::GpuMem1 => Self::run_gpu(context, true, par::config::V1, task),
                    cmd::Mode::GpuMem2 => Self::run_gpu(context, true, par::config::V2, task),
                    cmd::Mode::GpuMem3 => Self::run_gpu(context, true, par::config::V3, task),
                },
            ));
            info!("[{mode:?}] Completed execution");
        }

        let random_indices = {
            use rand::prelude::*;
            let mut rng = rand::thread_rng();
            (0..random_checks)
                .map(|_| {
                    (
                        rng.gen_range(0..task.matrices().len()),
                        rng.gen_range(0..task.n()),
                        rng.gen_range(0..task.n()),
                    )
                })
                .collect::<Vec<_>>()
        };

        use comfy_table::{presets::UTF8_FULL, Cell, CellAlignment, Table};
        let mut table = Table::new();
        table.load_preset(UTF8_FULL).set_header(
            ["Mode", "Time"]
                .iter()
                .map(|s| s.to_string())
                .chain(
                    random_indices
                        .iter()
                        .map(|(index, x, y)| format!("[{index}]({x}, {y})")),
                )
                .collect::<Vec<_>>(),
        );
        for (mode, verdict) in verdicts {
            let mode = Cell::new(format!("{mode:?}")).set_alignment(CellAlignment::Right);
            if let Some(Verdict { solution, time }) = verdict {
                table.add_row([mode, Cell::new(format!("{time:?}"))].into_iter().chain(
                    random_indices.iter().map(|(index, x, y)| {
                        Cell::new(format!("{:.6}", &solution.0[*index][*x + *y * task.n()]))
                            .set_alignment(CellAlignment::Right)
                    }),
                ));
            } else {
                table.add_row([mode, Cell::new("-").fg(Color::Grey)]);
            }
        }
        println!("{table}");
    }

    fn run_cpu(parallel: bool, task: &Task) -> Verdict {
        if parallel {
            Self::measure(|| seq::solve_par(task))
        } else {
            Self::measure(|| seq::solve(task))
        }
    }

    fn run_gpu(
        context: &Context,
        memoizing: bool,
        config: par::config::Config,
        task: &Task,
    ) -> Option<Verdict> {
        let mut executor = match par::Executor::new(task.n(), context, config) {
            Ok(executor) => executor,
            Err(cause) => {
                warn!("Unable to run execution: {cause}");
                return None;
            }
        };

        Some(if memoizing {
            Self::measure(|| executor.solve_memoizing(task))
        } else {
            Self::measure(|| executor.solve(task))
        })
    }

    #[inline(always)]
    fn measure(job: impl FnOnce() -> Solution) -> Verdict {
        let begin = Instant::now();
        let solution = job();
        let end = Instant::now();

        Verdict {
            solution,
            time: end - begin,
        }
    }
}

struct Verdict {
    solution: Solution,
    time: Duration,
}

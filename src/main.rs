use opencl3::{
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
};
use task::Matrix;

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

    let a = Matrix::<4>::zeros();
    let b = Matrix::<4>::ones();

    let c_seq = seq::multiply(a.clone(), b.clone());
    println!("C (sequential) = {c_seq:?}");
    let c_par = par::multiply(context, a, b);
    println!("C (parallel) = {c_par:?}");
}

use opencl3::platform::get_platforms;
use task::Matrix;

mod cmd;
mod par;
mod seq;
mod task;

fn main() {
    let platforms = get_platforms().unwrap();

    for platform in platforms {
        println!("Platform {platform:?}");
        let devices = platform
            .get_devices(cl3::device::CL_DEVICE_TYPE_GPU)
            .unwrap();
        for device in devices {
            println!("\t* Device: {device:?}");
        }
    }

    let a = Matrix::<8>::zeros();
    let b = Matrix::<8>::ones();

    let c_seq = seq::multiply(a, b);
    // let c_par = par::multiply(a, b);
    println!("C = {c_seq:?}");
}

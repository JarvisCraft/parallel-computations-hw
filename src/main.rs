use opencl3::platform::get_platforms;

mod cmd;

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
}

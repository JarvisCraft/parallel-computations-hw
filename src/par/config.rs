use std::num::NonZeroUsize;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub(super) src: &'static str,
    pub(super) work_size: WorkSize,
}

#[derive(Debug, Clone, Copy)]
pub struct WorkSize {
    pub(super) local: Option<NonZeroUsize>,
    pub(super) per_thread: NonZeroUsize,
}

pub const V1: Config = Config {
    src: include_str!("multiply1.cl"),
    work_size: WorkSize {
        local: None,
        per_thread: NonZeroUsize::new(1).unwrap(),
    },
};
pub const V2: Config = Config {
    src: include_str!("multiply2.cl"),
    work_size: WorkSize {
        local: Some(NonZeroUsize::new(8).unwrap()),
        per_thread: NonZeroUsize::new(1).unwrap(),
    },
};
pub const V3: Config = Config {
    src: include_str!("multiply3.cl"),
    work_size: WorkSize {
        local: Some(NonZeroUsize::new(16).unwrap()),
        per_thread: NonZeroUsize::new(4).unwrap(),
    },
};

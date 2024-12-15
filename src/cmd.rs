use std::str::FromStr;

use clap::Parser;

#[derive(Parser, Debug)]
pub struct Cmd {
    /// Modes in which the computation performs
    #[arg(long, short, default_values = ["1", "2", "3", "4", "5", "6", "7", "8"])]
    pub modes: Vec<Mode>,
    /// The number of matrices
    #[arg(long, short = 'N')]
    pub matrices: usize,
    /// Square matrix dimension
    #[arg(long, short = 'n')]
    pub dimension: usize,
    #[arg(long, short)]
    pub sample: bool,
    /// The number of random indices checked in each of the result matrices
    #[arg(long, short, default_value_t = 4)]
    pub checked_indices: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    CpuSingleThreaded,
    CpuMultiThreaded,
    GpuNaive1,
    GpuNaive2,
    GpuNaive3,
    GpuMem1,
    GpuMem2,
    GpuMem3,
}

impl FromStr for Mode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_ascii_lowercase().as_ref() {
            "1" | "c1" => Self::CpuSingleThreaded,
            "2" | "cm" => Self::CpuMultiThreaded,
            "3" | "gn1" => Self::GpuNaive1,
            "4" | "gn2" => Self::GpuNaive2,
            "5" | "gn3" => Self::GpuNaive3,
            "6" | "gm1" => Self::GpuMem1,
            "7" | "gm2" => Self::GpuMem2,
            "8" | "gm3" => Self::GpuMem3,
            _ => return Err(format!("Unknown mode {s:?}")),
        })
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn parses() {
        Cmd::command().debug_assert();
    }
}

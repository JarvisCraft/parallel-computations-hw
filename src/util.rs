//  Based on yet unstable https://github.com/rust-lang/rust/pull/116176.
pub fn sqrt(mut value: usize) -> usize {
    if value < 2 {
        return value;
    }

    let mut res = 0;
    let mut one = 1 << (value.ilog2() & !1);

    while one != 0 {
        if value >= res + one {
            value -= res + one;
            res = (res >> 1) + one;
        } else {
            res >>= 1;
        }
        one >>= 2;
    }

    res
}

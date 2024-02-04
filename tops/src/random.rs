use libc::c_ulonglong;

extern "C" {
    fn cuda_reset_random_seed(seed: c_ulonglong);
}

pub fn reset_random_seed(seed: u64) {
    unsafe {
        cuda_reset_random_seed(seed);
    }
}

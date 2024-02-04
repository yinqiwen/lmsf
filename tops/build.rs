use cmake::Config;

fn main() {
    let dst = Config::new("csrc")
        .define("CMAKE_CUDA_ARCHITECTURES", "75")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();
    let lib_dir = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=tops");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

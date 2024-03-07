use std::ops::Deref;

use candle_core::{
    cuda_backend::cudarc::driver::{
        result::{memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async},
        CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig,
    },
    CpuStorage, CudaDevice, DType, Device, IndexOp, Layout, Shape, Storage, Tensor, D,
};
use candle_ext::TensorExt;

struct ArgSort;
impl candle_core::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "arg-sort"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        if layout.shape().rank() != 1 {
            candle_core::bail!(
                "input should have a single dimension, got {:?}",
                layout.shape()
            )
        }
        let slice = storage.as_slice::<f32>()?;
        let src = match layout.contiguous_offsets() {
            None => candle_core::bail!("input has to be contiguous"),
            Some((o1, o2)) => &slice[o1..o2],
        };
        let mut dst = (0..src.len() as u32).collect::<Vec<u32>>();
        dst.sort_by(|&i, &j| src[i as usize].total_cmp(&src[j as usize]));
        let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, layout.shape().clone()))
    }
}

#[test]
fn test_index0() -> candle_core::Result<()> {
    let a = Tensor::new(&[0.0f32, 1.0, 3.0, 2.0, -12.0, 4.0, 3.5, 8.0], &Device::Cpu)?;
    let v = a.i(3)?.to_scalar::<f32>()?;
    println!("v={}", v);

    let a = a.reshape((2, 4))?;
    let b = a.i((1, 2..))?;
    println!("a={:?}, b={:?}", a.shape(), b.shape());
    let v = b.to_vec1::<f32>()?;
    println!("v{:?}", v);
    Ok(())
}

#[test]
fn test_sort() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let (before_free, _) =
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info().unwrap();
    println!("before_free:{}", before_free);
    let a = Tensor::randn(0f32, 1., (1, 32000), &device)?.to_dtype(DType::F16)?;
    let v = a.cumsum(D::Minus1)?;
    let (after_free, _) =
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info().unwrap();
    println!(
        "v shape:{:?}, cumsum use gpu mem:{}",
        v.shape(),
        before_free - after_free,
    );
    drop(a);
    drop(v);

    let (after_drop, _) =
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info().unwrap();
    println!("afrer drop all, use gpu mem:{}", before_free - after_drop,);
    Ok(())
}

#[test]
fn test_cumsum() -> candle_core::Result<()> {
    let a = Tensor::new(&[0.0f32, 1.0, 3.0, 2.0, -12.0, 4.0, 3.5], &Device::Cpu)?;

    let indices = a.apply_op1(ArgSort)?;
    let a_sorted = a.gather(&indices, 0)?;
    println!("{indices}");
    println!("{a_sorted}");
    Ok(())
}

#[test]
fn test_repeate() -> candle_core::Result<()> {
    let a = Tensor::new(&[1.0], &Device::Cpu)?;
    let a = a.reshape((1, 1))?;
    let start = std::time::Instant::now();
    let a = a.repeat((1, 32000))?;
    println!(
        "cpu tensor repeat to {:?} cost {:?}",
        a.shape(),
        start.elapsed()
    );

    let device = Device::new_cuda(0)?;

    let b = Tensor::new(&[1.0], &device)?;
    let b = b.reshape((1, 1))?;
    let start = std::time::Instant::now();
    let b = b.repeat((1, 32000))?;
    println!(
        "cuda tensor repeat to {:?} cost {:?}",
        b.shape(),
        start.elapsed()
    );
    Ok(())
}

#[test]
fn test_tesnor_dims() {
    let v = (0..100 as i64).collect::<Vec<_>>();
    let dev = candle_core::Device::Cpu;
    let tensor = Tensor::from_vec(v, (2, 5, 10), &dev).unwrap();
    assert_eq!(2, tensor.shape().dims()[0]);
    assert_eq!(5, tensor.shape().dims()[1]);
    assert_eq!(10, tensor.shape().dims()[2]);
}

#[test]
fn test_empty_tesnor() {
    let v: Vec<i64> = Vec::new();
    let dev = candle_core::Device::Cpu;
    let tensor = Tensor::from_vec(v, (1, 0), &dev).unwrap();
    assert_eq!(1, tensor.shape().dims()[0]);
    assert_eq!(0, tensor.shape().dims()[1]);
}

#[test]
fn test_tesnor_cat() -> candle_core::Result<()> {
    let device = Device::new_cuda(0)?;
    let cuda_dev = match &device {
        Device::Cuda(c) => c,
        _ => {
            candle_core::bail!("unexpected")
        }
    };
    let (before_free, _) =
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info().unwrap();
    println!("before_free:{}", before_free);
    let a = Tensor::zeros((32, 1024, 1024), DType::F32, &device)?;
    let b = Tensor::zeros((32, 1024, 1024), DType::F32, &device)?;
    cuda_dev.synchronize();
    let (after_free, _) =
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info().unwrap();
    println!("Create 2 tensors use gpu mem:{}", before_free - after_free,);
    let c = Tensor::cat(&[&a, &b], 0)?;
    cuda_dev.synchronize();
    let (after_free, _) =
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info().unwrap();
    println!(
        "After cat 2 tensors use gpu mem:{}",
        before_free - after_free,
    );
    drop(a);
    drop(b);
    //cuda_dev.synchronize();
    let (after_free, _) =
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info().unwrap();
    println!(
        "After drop 2 tensors use gpu mem:{}",
        before_free - after_free,
    );
    Ok(())
}

#[test]
fn test_create_tesnor() -> candle_core::Result<()> {
    let v: Vec<f32> = vec![0.8, 0.9];
    let dev = candle_core::Device::Cpu;
    let start = std::time::Instant::now();
    let tensor = Tensor::new(v, &dev).unwrap();
    println!(
        "cost {:?} to create cpu tensor:{:?}",
        start.elapsed(),
        tensor.shape()
    );

    let cuda_dev = Device::new_cuda(0)?;
    let v: Vec<f32> = vec![0.8, 1.2, 3.4];
    let start = std::time::Instant::now();
    let tensor = Tensor::new(v, &cuda_dev).unwrap();
    println!("cost {:?} to create cuda tensor", start.elapsed());
    Ok(())
}

#[test]
fn test_tensor_slice() -> candle_core::Result<()> {
    let cuda_dev = Device::new_cuda(0)?;
    let t = Tensor::ones((2, 1, 12288), DType::F32, &cuda_dev)?;
    let t1 = t.i((.., .., 0..4096))?;
    let t2 = t.i((.., .., 4096..8192))?;
    let t3 = t.i((.., .., 8192..))?;
    println!(
        "Shapes: {:?}/{:?}/{:?}/{:?}",
        t.shape(),
        t1.shape(),
        t2.shape(),
        t3.shape()
    );
    println!(
        "Strides: {:?}/{:?}/{:?}/{:?}",
        t.stride(),
        t1.stride(),
        t2.stride(),
        t3.stride()
    );
    Ok(())
}

#[test]
fn test_tesnor_or() {
    let dev = candle_core::Device::Cpu;
    let data = vec![1_i64, 2, 3, 4, 5, 6, 7, 8, 9];
    let t = Tensor::from_vec(data, (3, 3), &dev).unwrap();
    let t1 = t.eq(5_i64).unwrap();
    let t2 = t.gt(3_i64).unwrap();
    // let cond = t1.add(&t2).unwrap();
    let cond = t1.where_cond(&t1, &t2).unwrap();
    let tmp0 = Tensor::zeros_like(&cond).unwrap();
    let tmp1 = Tensor::ones_like(&cond).unwrap();
    let cond1 = cond.where_cond(&tmp0, &tmp1).unwrap();
    // print!("{:?}", t1.to_string());
    // print!("{:?}", t2.to_string());
    // print!("{:?}", cond.to_string());
    let x_data = vec![100_i64, 101, 102, 103, 104, 105, 106, 107, 108];
    let y_data = vec![1000_i64, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008];
    let x = Tensor::from_vec(x_data, (3, 3), &dev).unwrap();
    let y = Tensor::from_vec(y_data, (3, 3), &dev).unwrap();
    let z = cond.where_cond(&x, &y).unwrap();
    println!("{:?}", cond.to_string());
    println!("{:?}", cond1.to_string());
    println!("{:?}", z.to_string());
}

#[test]
fn test_tesnor_reshape() {
    let dev = candle_core::Device::Cpu;
    let t = Tensor::zeros((1, 1, 4096), DType::I64, &dev).unwrap();
    let t1 = t.reshape(((), 32, 128)).unwrap();
    print!("{:?}", t1.shape());
}

#[test]
fn test_cuda_kernels_launc() -> candle_core::Result<()> {
    let device = candle_core::Device::new_cuda(0).unwrap();
    let cuda_device = if let candle_core::Device::Cuda(cuda_dev) = &device {
        cuda_dev
    } else {
        unimplemented!("unreach");
    };
    // let func = cuda_device
    //     .get_or_load_func("test_cuda0", kernels::TEST)
    //     .unwrap();
    // let vec = vec![123 as i64];
    // let tensor = Tensor::from_vec(vec, (1,), &device).unwrap();

    // let null_data = cuda_device.null::<i64>().unwrap();
    // let (query_storage, _) = tensor.storage_and_layout();
    // let query_data = match query_storage.deref() {
    //     Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<i64>()?,
    //     _ => unreachable!("unexpected storage type"),
    // };
    // let mut cfg = LaunchConfig {
    //     grid_dim: (1, 1, 1),
    //     block_dim: (1, 1, 1),
    //     shared_mem_bytes: 0,
    // };
    // let params = (query_data,);
    // unsafe {
    //     func.clone().launch(cfg, params).unwrap();
    // }
    // cuda_device.synchronize();
    Ok(())
}

#[test]
fn test_broadcast_sub() -> candle_core::Result<()> {
    let device = candle_core::Device::new_cuda(0).unwrap();
    let top_k_mask = Tensor::new(32000_i64, &device)?;
    let k = Tensor::new(&[5_i64, 5], &device)?;
    let top_k_mask = top_k_mask.broadcast_sub(&k)?;
    println!("{}", top_k_mask.to_string());

    let test = Tensor::rand(1_f32, 10.0, (2, 10), &device)?;
    println!("{}", test.to_string());
    let test2 = test.i((.., 9..10))?;
    println!("{:?} {:?}", test2.shape(), test2.stride());
    tops::unsafe_tensor_zero(&test2)?;
    println!("{}", test.to_string());

    // println!("{} {:?}", test.to_string(), test.stride());
    // let src = Tensor::arange(0u32, 2, &device)?
    //     .reshape((2, 1))?
    //     .to_dtype(DType::F32)?;
    // let test2 = test.slice_assign(&[0..2, 7..8], &src)?;
    // println!("{} {:?}", test.to_string(), test.stride());
    // println!("{} {:?}", test2.to_string(), test2.stride());
    // // let last = test.i((.., 7))?;
    // println!(
    //     "{} {:?} {:?}",
    //     last.to_string(),
    //     last.shape(),
    //     last.stride()
    // );
    Ok(())
}

#[repr(C)]
struct TestStruct {
    a: i32,
    b: i32,
    c: i32,
}

unsafe impl DeviceRepr for TestStruct {}
// #[test]
// fn test_struct_cuda_kernels_launc() -> candle_core::Result<()> {
//     let device = candle_core::Device::new_cuda(0).unwrap();
//     let cuda_device = if let candle_core::Device::Cuda(cuda_dev) = &device {
//         cuda_dev
//     } else {
//         unimplemented!("unreach");
//     };
//     let func = cuda_device
//         .get_or_load_func("test_cuda1", kernels::TEST)
//         .unwrap();

//     let thing = TestStruct { a: 1, b: 2, c: 3 };
//     let cfg = LaunchConfig {
//         grid_dim: (1, 1, 1),
//         block_dim: (1, 1, 1),
//         shared_mem_bytes: 0,
//     };
//     let params = (thing,);
//     unsafe {
//         func.clone().launch(cfg, params).unwrap();
//     }
//     cuda_device.synchronize();
//     Ok(())
// }

fn test_x(arg: Option<&str>) {
    let x = if let Some(v) = arg { v } else { "" };

    // let s = match x {
    //     "1" => 1,
    //     "2" => "2",
    // };
}

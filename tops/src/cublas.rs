use candle::{
    cuda_backend::{cudarc::driver::sys::CUstream, DeviceId},
    DType, Device, Shape, Tensor,
};
use common::ffi::{get_scalar_type, CShapeView, CTensorView, ScalarType};
use common::{DefaultTensorCreator, TensorCreator};
use libc::{c_int, c_void};

extern "C" {
    fn new_gemm(device: c_int, stream: CUstream, dtype: c_int) -> *mut c_void;
    fn delete_gemm(gemm: *mut c_void);
    fn save_gemm_config();
    fn gemm_config(
        gemm: *mut c_void,
        dtype: c_int,
        transA: bool,
        transB: bool,
        min_input: CShapeView,
        max_input: CShapeView,
        weight: CShapeView,
    );
    fn gemm_execute(
        gemm: *mut c_void,
        transa: c_int,
        transb: c_int,
        A: CTensorView,
        B: CTensorView,
        C: CTensorView,
    ) -> c_int;
}

pub struct CublasWrapper {
    wrapper: *mut c_void,
}
unsafe impl Send for CublasWrapper {}
impl Drop for CublasWrapper {
    fn drop(&mut self) {
        unsafe {
            delete_gemm(self.wrapper);
        }
        self.wrapper = std::ptr::null_mut();
    }
}

impl CublasWrapper {
    pub fn new(device: &Device, dtype: DType, stream: CUstream) -> candle::Result<CublasWrapper> {
        let device_id = match device {
            Device::Cuda(cuda) => *cuda.cu_device(),
            _ => {
                candle::bail!("not supported device")
            }
        };

        let p = match dtype {
            DType::F16 => unsafe { new_gemm(device_id, stream, ScalarType::DATA_F16 as i32) },
            DType::BF16 => unsafe { new_gemm(device_id, stream, ScalarType::DATA_BF16 as i32) },
            DType::F32 => unsafe { new_gemm(device_id, stream, ScalarType::DATA_F32 as i32) },
            _ => {
                candle::bail!("not supported dtype")
            }
        };
        Ok(Self { wrapper: p })
    }
    pub fn linear_<F: TensorCreator>(
        &self,
        input: &Tensor,
        weight: &Tensor,
        tensor_creator: &mut F,
    ) -> candle::Result<Tensor> {
        let output = if input.dims().len() == 3 {
            let (batch, num, tmp) = input.dims3()?;
            let (output_dims, _) = weight.dims2()?;
            //let output = Tensor::zeros((batch, num, output_dims), input.dtype(), input.device())?;
            tensor_creator.new(
                (batch, num, output_dims),
                input.dtype(),
                input.device(),
                false,
            )?
        } else {
            let (num, tmp) = input.dims2()?;
            let (output_dims, _) = weight.dims2()?;
            //let output = Tensor::zeros((batch, num, output_dims), input.dtype(), input.device())?;
            tensor_creator.new((num, output_dims), input.dtype(), input.device(), false)?
        };

        let input_view = CTensorView::from(input, false)?;
        let weight_view = CTensorView::from(weight, false)?;
        let output_view = CTensorView::from(&output, false)?;

        // println!("#####AA");
        let rc = unsafe { gemm_execute(self.wrapper, 0, 1, input_view, weight_view, output_view) };
        // println!("#####BB");
        if 0 != rc {
            // println!(
            //     "##linear {:?},  {:?} {:?}",
            //     input.shape(),
            //     weight.shape(),
            //     input.dtype()
            // );

            // unsafe {
            //     gemm_config(
            //         self.wrapper,
            //         get_scalar_type(input.dtype()) as c_int,
            //         false,
            //         true,
            //         CShapeView::new(&min_input_shape),
            //         CShapeView::new(&max_input_shape),
            //         CShapeView::new(weight.shape()),
            //     );
            // }
            // println!("#####B");
        }

        Ok(output)
    }
    pub fn linear(&self, input: &Tensor, weight: &Tensor) -> candle::Result<Tensor> {
        let mut default_creator = DefaultTensorCreator {};
        self.linear_(input, weight, &mut default_creator)
    }
}

#[test]
fn test_gemm_config() -> candle::Result<()> {
    unsafe {
        let gemm = new_gemm(0, std::ptr::null_mut(), ScalarType::DATA_F16 as i32);

        let mut configs = Vec::new();

        let config = (
            Shape::from_dims(&[1, 1, 11008]),
            Shape::from_dims(&[8, 1, 11008]),
            Shape::from_dims(&[4096, 11008]),
        );
        configs.push(config);

        let config = (
            Shape::from_dims(&[1, 1, 4096]),
            Shape::from_dims(&[8, 1, 4096]),
            Shape::from_dims(&[12288, 4096]),
        );
        configs.push(config);

        let config = (
            Shape::from_dims(&[1, 1, 4096]),
            Shape::from_dims(&[8, 1, 4096]),
            Shape::from_dims(&[4096, 4096]),
        );
        configs.push(config);

        let config = (
            Shape::from_dims(&[1, 1, 11008]),
            Shape::from_dims(&[8, 1, 11008]),
            Shape::from_dims(&[4096, 11008]),
        );
        configs.push(config);

        let config = (
            Shape::from_dims(&[1, 1, 4096]),
            Shape::from_dims(&[8, 1, 4096]),
            Shape::from_dims(&[22016, 4096]),
        );
        configs.push(config);

        let config = (
            Shape::from_dims(&[1, 4096]),
            Shape::from_dims(&[8, 4096]),
            Shape::from_dims(&[32000, 4096]),
        );
        configs.push(config);

        let config = (
            Shape::from_dims(&[1, 9, 4096]),
            Shape::from_dims(&[8, 9, 4096]),
            Shape::from_dims(&[12288, 4096]),
        );
        configs.push(config);

        let config = (
            Shape::from_dims(&[1, 9, 4096]),
            Shape::from_dims(&[8, 9, 4096]),
            Shape::from_dims(&[4096, 4096]),
        );
        configs.push(config);

        let config = (
            Shape::from_dims(&[1, 9, 4096]),
            Shape::from_dims(&[8, 9, 4096]),
            Shape::from_dims(&[22016, 4096]),
        );
        configs.push(config);

        let config = (
            Shape::from_dims(&[1, 9, 4096]),
            Shape::from_dims(&[8, 9, 4096]),
            Shape::from_dims(&[12288, 4096]),
        );
        configs.push(config);

        let config = (
            Shape::from_dims(&[1, 9, 11008]),
            Shape::from_dims(&[8, 9, 11008]),
            Shape::from_dims(&[4096, 11008]),
        );
        configs.push(config);

        for (min_input, max_input, weight) in configs {
            gemm_config(
                gemm,
                get_scalar_type(DType::F16) as c_int,
                false,
                true,
                CShapeView::new(&min_input),
                CShapeView::new(&max_input),
                CShapeView::new(&weight),
            );
        }

        save_gemm_config();
    }

    Ok(())
}

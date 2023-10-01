use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor};
use orion::numbers::{FixedTrait, FP16x16};


fn dense_4_bias() -> Tensor<FP16x16> {
    let mut shape = array![3,].span();
    let mut data = array![
FP16x16 { mag: 35365, sign: true },FP16x16 { mag: 15187, sign: false },FP16x16 { mag: 78, sign: true },].span();
    Tensor {shape: shape, data: data}
}

use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor};
use orion::numbers::{FixedTrait, FP16x16};


fn dense_4_bias() -> Tensor<FP16x16> {
    let mut shape = array![3,].span();
    let mut data = array![
FP16x16 { mag: 34829, sign: true },FP16x16 { mag: 17676, sign: false },FP16x16 { mag: 6040, sign: true },].span();
    TensorTrait::new(shape, data)
}

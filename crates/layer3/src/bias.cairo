use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor};
use orion::numbers::{FixedTrait, FP16x16};


fn dense_3_bias() -> Tensor<FP16x16> {
    let mut shape = array![25,].span();
    let mut data = array![
FP16x16 { mag: 41273, sign: true },FP16x16 { mag: 9687, sign: false },FP16x16 { mag: 32712, sign: true },FP16x16 { mag: 26360, sign: false },FP16x16 { mag: 51419, sign: false },FP16x16 { mag: 29840, sign: false },FP16x16 { mag: 35505, sign: true },FP16x16 { mag: 23766, sign: false },FP16x16 { mag: 20915, sign: false },FP16x16 { mag: 33433, sign: false },FP16x16 { mag: 24130, sign: false },FP16x16 { mag: 34914, sign: false },FP16x16 { mag: 36136, sign: true },FP16x16 { mag: 7546, sign: true },FP16x16 { mag: 37949, sign: false },FP16x16 { mag: 3854, sign: false },FP16x16 { mag: 87616, sign: true },FP16x16 { mag: 17797, sign: false },FP16x16 { mag: 35175, sign: false },FP16x16 { mag: 23402, sign: true },FP16x16 { mag: 10936, sign: true },FP16x16 { mag: 24280, sign: false },FP16x16 { mag: 49339, sign: false },FP16x16 { mag: 14233, sign: true },FP16x16 { mag: 31451, sign: false },].span();
    Tensor {shape: shape, data: data}
}

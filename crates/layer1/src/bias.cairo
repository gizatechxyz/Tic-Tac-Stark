use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor};
use orion::numbers::{FixedTrait, FP16x16};


fn dense_1_bias() -> Tensor<FP16x16> {
    let mut shape = array![50,].span();
    let mut data = array![
FP16x16 { mag: 3637, sign: true },FP16x16 { mag: 65217, sign: true },FP16x16 { mag: 30006, sign: true },FP16x16 { mag: 13709, sign: true },FP16x16 { mag: 32719, sign: true },FP16x16 { mag: 14716, sign: true },FP16x16 { mag: 45660, sign: true },FP16x16 { mag: 33738, sign: true },FP16x16 { mag: 4408, sign: true },FP16x16 { mag: 101945, sign: true },FP16x16 { mag: 46777, sign: true },FP16x16 { mag: 17729, sign: true },FP16x16 { mag: 24198, sign: true },FP16x16 { mag: 143689, sign: false },FP16x16 { mag: 51954, sign: true },FP16x16 { mag: 39161, sign: true },FP16x16 { mag: 139196, sign: false },FP16x16 { mag: 10935, sign: true },FP16x16 { mag: 62698, sign: true },FP16x16 { mag: 15699, sign: true },FP16x16 { mag: 35233, sign: true },FP16x16 { mag: 9012, sign: true },FP16x16 { mag: 6727, sign: true },FP16x16 { mag: 11052, sign: true },FP16x16 { mag: 31303, sign: true },FP16x16 { mag: 47611, sign: true },FP16x16 { mag: 20562, sign: false },FP16x16 { mag: 26690, sign: true },FP16x16 { mag: 112661, sign: true },FP16x16 { mag: 111687, sign: true },FP16x16 { mag: 105390, sign: false },FP16x16 { mag: 41860, sign: true },FP16x16 { mag: 38713, sign: true },FP16x16 { mag: 28252, sign: false },FP16x16 { mag: 6000, sign: false },FP16x16 { mag: 146370, sign: true },FP16x16 { mag: 85398, sign: true },FP16x16 { mag: 70642, sign: true },FP16x16 { mag: 94607, sign: true },FP16x16 { mag: 77093, sign: false },FP16x16 { mag: 87907, sign: true },FP16x16 { mag: 47120, sign: true },FP16x16 { mag: 28634, sign: true },FP16x16 { mag: 191224, sign: true },FP16x16 { mag: 3433, sign: true },FP16x16 { mag: 33516, sign: true },FP16x16 { mag: 39638, sign: true },FP16x16 { mag: 2118, sign: false },FP16x16 { mag: 36280, sign: true },FP16x16 { mag: 260, sign: true },].span();
    TensorTrait::new(shape, data)
}

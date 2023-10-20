use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor, FP16x16Tensor};
use orion::numbers::{FixedTrait, FP16x16};


fn dense_3_bias() -> Tensor<FP16x16> {
    let mut shape = array![25,].span();
    let mut data = array![
FP16x16 { mag: 25692, sign: false },FP16x16 { mag: 3057, sign: true },FP16x16 { mag: 25250, sign: true },FP16x16 { mag: 34867, sign: false },FP16x16 { mag: 9890, sign: true },FP16x16 { mag: 29844, sign: true },FP16x16 { mag: 11660, sign: false },FP16x16 { mag: 2011, sign: true },FP16x16 { mag: 43202, sign: false },FP16x16 { mag: 51364, sign: true },FP16x16 { mag: 2691, sign: true },FP16x16 { mag: 3300, sign: false },FP16x16 { mag: 5803, sign: false },FP16x16 { mag: 61134, sign: true },FP16x16 { mag: 42118, sign: false },FP16x16 { mag: 20747, sign: false },FP16x16 { mag: 41008, sign: false },FP16x16 { mag: 33208, sign: true },FP16x16 { mag: 71749, sign: true },FP16x16 { mag: 47177, sign: false },FP16x16 { mag: 2332, sign: true },FP16x16 { mag: 11433, sign: false },FP16x16 { mag: 14137, sign: false },FP16x16 { mag: 86404, sign: true },FP16x16 { mag: 13087, sign: true },].span();
    TensorTrait::new(shape, data)
}

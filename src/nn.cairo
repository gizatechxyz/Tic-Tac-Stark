use orion::operators::tensor::{Tensor, FP16x16Tensor};
use orion::operators::nn::{FP16x16NN, NNTrait};
use orion::numbers::FP16x16;

fn dense_relu(i: Tensor<FP16x16>, w: Tensor<FP16x16>, b: Tensor<FP16x16>) -> Tensor<FP16x16> {
    let x = NNTrait::linear(i, w, b);
    NNTrait::relu(@x)
}

fn dense_softmax(i: Tensor<FP16x16>, w: Tensor<FP16x16>, b: Tensor<FP16x16>) -> Tensor<FP16x16> {
    let x = NNTrait::linear(i, w, b);
    NNTrait::softmax(@x, 0)
}

use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::FP16x16;

use tic_tac_toe::nn::{dense_relu, dense_softmax};

use layer0::weights::dense_weights as w0;
use layer1::weights::dense_1_weights as w1;
// use layer2::weights::dense_2_weights as w2;
// use layer3::weights::dense_3_weights as w3;
// use layer4::weights::dense_4_weights as w4;

use layer0::bias::dense_bias as b0;
use layer1::bias::dense_1_bias as b1;
// use layer2::bias::dense_2_bias as b2;
// use layer3::bias::dense_3_bias as b3;
// use layer4::bias::dense_4_bias as b4;

#[test]
#[available_gas(20000000000000)]
fn test_ttt() {
    let zero = FP16x16 { mag: 0, sign: false };

    let mut x = TensorTrait::new(
        array![9].span(), array![zero, zero, zero, zero, zero, zero, zero, zero, zero].span()
    );

    x = dense_relu(x, w0(), b0());
    x = dense_relu(x, w1(), b1());
// x = dense_relu(x, w2(), b2());
// x = dense_relu(x, w3(), b3());
// x = dense_softmax(x, w4(), b4());

}


use layer0::weights::dense_weights;
use layer1::weights::dense_1_weights;
use layer2::weights::dense_2_weights;
use layer3::weights::dense_3_weights;
use layer4::weights::dense_4_weights;

use layer0::bias::dense_bias;
use layer1::bias::dense_1_bias;
use layer2::bias::dense_2_bias;
use layer3::bias::dense_3_bias;
use layer4::bias::dense_4_bias;

fn foo() {
    let w0 = dense_weights();
    let w1 = dense_1_weights();
    let w2 = dense_2_weights();
    let w3 = dense_3_weights();
    let w4 = dense_4_weights();

    let b0 = dense_bias();
    let b1 = dense_1_bias();
    let b2 = dense_2_bias();
    let b3 = dense_3_bias();
    let b4 = dense_4_bias();
}

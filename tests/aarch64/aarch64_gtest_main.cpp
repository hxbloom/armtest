#include <stdio.h>

#include "gtest/gtest.h"

#include "test_mat_fill.h"
#include "test_mat_substract_mean_normalize.h"
#include "test_eltwise_arm.h"
#include "test_relu_arm.h"
#include "test_absval_arm.h"
#include "test_batchnorm_arm.h"
#include "test_slice_arm.h"
#include "test_pooling22_arm.h"
#include "test_prelu_arm.h"
#include "test_pooling33_arm.h"
#include "test_conv22_arm.h"
#include "test_conv44_arm.h"
#include "test_conv33dw_arm.h"
#include "test_innerproduct_arm.h"
#include "test_conv55_arm.h"

int main(int argc, char **argv){
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

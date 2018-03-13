#pragma once
#include <arm_neon.h>
#include "gtest/gtest.h"
#include "math.h"


TEST(aarch64_assembly, prelu)
{
  int nn = 5;
  int nn2 = 5;
  const float slope = 0.01;
  
  float c0[20] __attribute__((aligned(16)))  = {-5,-4,-3,-2,-1,0,1,2,3,4,4,3,2,1,0,-1,-2,-3,-4,-5};
  float d0[20] __attribute__((aligned(16)))  = {-5,-4,-3,-2,-1,0,1,2,3,4,4,3,2,1,0,-1,-2,-3,-4,-5};
  
  float e0[20] __attribute__((aligned(16)))  = {-5,-4,-3,-2,-1,0,1,2,3,4,4,3,2,1,0,-1,-2,-3,-4,-5};
  
  // For Product
  float *cc = c0;
  float *dd = d0;
  
  float *ptr = (float*)cc;  
  float *ptr2 = (float*)dd;
  

// ****** Product
// Line 48
        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _slope = vdupq_n_f32(slope);
        for (; nn2>0; nn2--)
        {
            float32x4_t _p = vld1q_f32(ptr2);
            uint32x4_t _lemask = vcleq_f32(_p, _zero);
            float32x4_t _ps = vmulq_f32(_p, _slope);
            _p = vbslq_f32(_lemask, _ps, _p);
            vst1q_f32(ptr2, _p);

            ptr2 += 4;
        }

// armv8 NEON simd EOR/BIT supports 8B&16B only,
// 16B and 4S are both 128bit long,
// Treat V registers here as 16B here instead of 4S for bit manipulation instructions.
        asm volatile(
            "eor        v1.16b, v0.16b, v0.16b \n"
            "dup        v2.4s, %w4             \n"
            "0:                                \n"
            "prfm       pldl1keep, [%1, #128]  \n"
            "ld1        {v0.4s}, [%1]          \n"
            "fcmle      v3.4s, v0.4s, v1.4s    \n"
            "fmul       v4.4s, v0.4s, v2.4s    \n"
            "bit        v0.16b, v4.16b, v3.16b \n"
            "subs       %w0, %w0, #1           \n"
            "st1        {v0.4s}, [%1], #16     \n"
            "bne        0b                     \n"
            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            : "0"(nn),
              "1"(ptr),
              "r"(slope)    // %4
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4"
        );

  
  for (int i=0; i<20; i++)
  {
    EXPECT_EQ(*(cc+i), *(dd+i));
    if (e0[i]>=0.f)
      EXPECT_EQ(*(cc+i), e0[i]);
    else
      EXPECT_EQ(*(cc+i), slope*e0[i]);
  }

}

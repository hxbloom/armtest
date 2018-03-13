#pragma once
#include <arm_neon.h>
#include "gtest/gtest.h"
#include "math.h"


TEST(aarch64_assembly, relu)
{
  int nn = 5;
  int nn2 = 5;
  
  float c0[20] __attribute__((aligned(16))) = {-5,-4,-3,-2,-1,0,1,2,3,4,4,3,2,1,0,-1,-2,-3,-4,-5};
  float d0[20] __attribute__((aligned(16))) = {-5,-4,-3,-2,-1,0,1,2,3,4,4,3,2,1,0,-1,-2,-3,-4,-5};
  
  float e0[20] __attribute__((aligned(16))) = {-5,-4,-3,-2,-1,0,1,2,3,4,4,3,2,1,0,-1,-2,-3,-4,-5};
  
  // For Product
  float *cc = c0;
  float *dd = d0;
  
  float *ptr = (float*)cc;
  float *ptr2 = (float*)dd;
  

// ****** Product
// Line 48
            float32x4_t _zero = vdupq_n_f32(0.f);
            for (; nn2>0; nn2--)
            {
                float32x4_t _p = vld1q_f32(ptr2);
                _p = vmaxq_f32(_p, _zero);
                vst1q_f32(ptr2, _p);

                ptr2 += 4;
            }

// armv8 NEON simd EOR supports 8B&16B only,
// 16B and 4S are both 128bit long,
// Treat v0,v1 here as 16B here instead of 4S.
            asm volatile(
                "eor        v1.16b, v0.16b, v0.16b\n"
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "ld1        {v0.4s}, [%1]         \n"
                "fmax       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%1], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr)
                : "cc", "memory", "v0", "v1"
            );

/*
            asm volatile(
                "veor       q1, q0, q0          \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vmax.f32   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr)
                : "cc", "memory", "q0", "q1"
            );
*/


  
  for (int i=0; i<20; i++)
  {
    EXPECT_EQ(*(cc+i), *(dd+i));
    EXPECT_EQ(*(cc+i), fmaxf(e0[i],0.f));
  }

}

#pragma once
#include <arm_neon.h>
#include "gtest/gtest.h"

TEST(aarch64_assembly, slice)
{
  int nn = 3;
  int nn2 = 3; 
  
  float c0[24] __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
  float d0[24] __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
  
  float e0[24] __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
  
  float *cc = c0;
  float *cc_out = new float[24]();
  float *dd = d0;
  float *dd_out = new float[24]();
  
  float *ptr = (float*)cc;
  float *outptr = (float*)cc_out;
  float *ptr2 = (float*)dd;
  float *outptr2 = (float*)dd_out;
  

// Line 60
// Intrinsic
        for (; nn2>0; nn2--)
        {
            float32x4_t _p = vld1q_f32(ptr2);
            float32x4_t _p2 = vld1q_f32(ptr2+4);
            vst1q_f32(outptr2, _p);
            vst1q_f32(outptr2+4, _p2);

            ptr2 += 8;
            outptr2 += 8;
        }

// Assembly 64

#if __ARM_NEON
#if __aarch64__

        asm volatile(
            "0:                                   \n"
            "prfm       pldl1keep, [%1, #256]     \n"
            "ld1        {v0.4s, v1.4s}, [%1], #32 \n"
            "subs       %w0, %w0, #1              \n"
            "st1        {v0.4s, v1.4s}, [%2], #32 \n"
            "bne        0b                        \n"
            : "=r"(nn),     // %0
              "=r"(ptr),    // %1
              "=r"(outptr)  // %2
            : "0"(nn),
              "1"(ptr),
              "2"(outptr)
            : "cc", "memory", "v0", "v1"
        );

#else // aarch 32
/*
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
//            "vld1.f32   {d0-d1}, [%1 :64]! \n"
//            "vld1.f32   {d2-d3}, [%1 :64]! \n"
            "vld1.f32   {d0-d3}, [%1 :128]! \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
//            "vst1.f32   {d2-d3}, [%2 :128]! \n"
//            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(ptr),    // %1
              "=r"(outptr)  // %2
            : "0"(nn),
              "1"(ptr),
              "2"(outptr)
//            : "cc", "memory", "q0"
            : "cc", "memory", "q0", "q1"
        );
*/
#endif // __aarch64__
#endif // __ARM_NEON

  
  
  for (int i=0; i<24; i++)
  {
    //printf("the %d of cc is : %f\n", i, *(cc_out+i));
    //printf("-----------the %d of dd is : %f\n", i, *(dd_out+i));
    EXPECT_EQ(*(cc+i), *(dd+i));
    EXPECT_EQ(*(cc+i), e0[i]);
  }



  delete []cc_out;
  delete []dd_out;
  
}

#pragma once
#include <arm_neon.h>
#include "gtest/gtest.h"


TEST(aarch64_assembly, innerproduct)
{
  // Set input to 24, to let remaining to be 0, nn to be 3;
  // Consider 1 channel case
  int nn = 24 >> 3;
  int nn2 = 24 >> 3; 
  
  float c0[24]  __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
  float w0[24]  __attribute__((aligned(16))) = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
  float c1[24]  __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
  float w1[24]  __attribute__((aligned(16))) = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};

  // Bias
  float b0 = 10000.f;
  float b1 = 10000.f;
  
  float sum = 0.f;
  float sum2 = 0.f;
  
  const float* w = w0;
  const float* m = c0;
  const float* w2 = w1;
  const float* m2 = c1;
  
  // Output
        float32x4_t _sum = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
        float32x4_t _sumb = vdupq_n_f32(0.f);
        float32x4_t _sum2b = vdupq_n_f32(0.f);
  
// Line 175
// Intrinsic
#if __aarch64__   
            sum2 = b1;
            for (; nn2>0; nn2--)
            {
                float32x4_t _m = vld1q_f32(m2);
                float32x4_t _w = vld1q_f32(w2);
                _sumb = vfmaq_f32(_sumb, _m, _w);

                _m = vld1q_f32(m2 + 4);
                _w = vld1q_f32(w2 + 4);
                _sum2b = vfmaq_f32(_sum2b, _m, _w);

                m2 += 8;
                w2 += 8;
            }
    
            _sumb = vaddq_f32(_sumb, _sum2b);
            sum2 += vaddvq_f32(_sumb);

#endif // __aarch64__


// Assembly 64

            sum = b0;
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "0:                                   \n"
                "prfm       pldl1keep, [%1, #256]     \n"
                "ld1        {v0.4s, v1.4s}, [%1], #32 \n"
                "prfm       pldl1keep, [%2, #256]     \n"
                "ld1        {v2.4s, v3.4s}, [%2], #32 \n"
                "fmla       %3.4s, v0.4s, v2.4s       \n"
                "subs       %w0, %w0, #1              \n"
                "fmla       %4.4s, v1.4s, v3.4s       \n"
                "bne        0b                        \n"
                : "=r"(nn),     // %0
                  "=r"(m),      // %1
                  "=r"(w),      // %2
                  "=w"(_sum),   // %3
                  "=w"(_sum2)   // %4
                : "0"(nn),
                  "1"(m),
                  "2"(w),
                  "3"(_sum),
                  "4"(_sum2)
                : "cc", "memory", "v0", "v1", "v2", "v3"
            );
            }
            _sum = vaddq_f32(_sum, _sum2);
            sum += vaddvq_f32(_sum);
#else

            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.f32   {d0-d3}, [%1 :128]! \n"
                "pld        [%2, #256]          \n"
                "vld1.f32   {d4-d7}, [%2]!      \n"
                "vmla.f32   %q3, q0, q2         \n"
                "subs       %0, #1              \n"
                "vmla.f32   %q4, q1, q3         \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(m),      // %1
                  "=r"(w),      // %2
                  "=w"(_sum),   // %3
                  "=w"(_sum2)   // %4
                : "0"(nn),
                  "1"(m),
                  "2"(w),
                  "3"(_sum),
                  "4"(_sum2)
                : "cc", "memory", "q0", "q1", "q2", "q3"
            );
            
            _sum = vaddq_f32(_sum, _sum2);
            float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
            _sumss = vpadd_f32(_sumss, _sumss);
            sum += vget_lane_f32(_sumss, 0);
            
#endif // __aarch64__
  
  
  //printf("sum 1 is : %f , sum 2 is : %f .\n",sum, sum2);
  EXPECT_EQ(sum, sum2);
  EXPECT_EQ(sum, 10600.f);
}

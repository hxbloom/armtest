#pragma once
#include <arm_neon.h>
#include "gtest/gtest.h"
#include "math.h"


TEST(aarch64_assembly, pooling22) 
{
  // Test the assembly part only
  // Set to 16 to let 'int remain = outw - (nn << 2);' in the original code to be 0
  const int w = 16;
  const int h = 16;
  const int outw = w/2;
  const int outh = h/2;
  const int tailstep = w - 2*outw + w;
  
  int nn = outw >> 2;
  int nn2 = outw >> 2;
  
  float c0[w*h] __attribute__((aligned(16)));
  for (int i=0; i<w*h; i++)
    c0[i] = i+1;
  
  const float *r0 = c0;
  const float *r1 = c0 + w;
  float *outptr = new float[outw*outh]();
  const float *r02 = c0;
  const float *r12 = c0 + w;
  float *outptr2 = new float[outw*outh]();

  
// Line 48
// Intrinsic
        for (int i = 0; i < outh; i++)
        {
            nn2 = outw >> 2;
            for (; nn2>0; nn2--)
            {
                float32x4_t _r00 = vld1q_f32(r02);
                float32x4_t _r10 = vld1q_f32(r12);
                float32x4_t _r01 = vld1q_f32(r02 + 4);
                float32x4_t _r11 = vld1q_f32(r12 + 4);

                float32x4_t _max0 = vmaxq_f32(_r00, _r10);
                float32x4_t _max1 = vmaxq_f32(_r01, _r11);

                float32x4_t _max = vpmaxq_f32(_max0, _max1);

                vst1q_f32(outptr2, _max);

                r02 += 8;
                r12 += 8;
                outptr2 += 4;
            }

            r02 += tailstep;
            r12 += tailstep;
        }
// Assembly 64
        for (int i = 0; i < outh; i++)
        {
            nn = outw >> 2;
            asm volatile(
                "0:                                   \n"
                "prfm       pldl1keep, [%1, #256]     \n"
                "prfm       pldl1keep, [%2, #256]     \n"
                "ld1        {v0.4s, v1.4s}, [%1], #32 \n"
                "ld1        {v2.4s, v3.4s}, [%2], #32 \n"
                "fmax       v0.4s, v0.4s, v2.4s       \n"
                "fmax       v1.4s, v1.4s, v3.4s       \n"
                "fmaxp      v2.4s, v0.4s, v1.4s       \n"
                "subs       %w0, %w0, #1              \n"
                "st1        {v2.4s}, [%3], #16        \n"
                "bne        0b                        \n"
                : "=r"(nn),     // %0
                  "=r"(r0),     // %1
                  "=r"(r1),     // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(r0),
                  "2"(r1),
                  "3"(outptr)
                : "cc", "memory", "v0", "v1", "v2", "v3"
            );
            
            r0 += tailstep;
            r1 += tailstep;
        }


  outptr -= outw*outh;
  outptr2 -= outw*outh;
  
  for (int i=0; i<outw*outh; i++)
  {
    EXPECT_NEAR(*(outptr+i), *(outptr2+i),1e-5);
    EXPECT_NEAR(*(outptr+i), w+2+2*(i%outw)+(i/outw)*2*w,1e-5);
  }



  delete []outptr;
  delete []outptr2;
  
}

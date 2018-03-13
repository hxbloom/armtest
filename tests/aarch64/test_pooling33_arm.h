#pragma once
#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>
#include "gtest/gtest.h"

TEST(aarch64_assembly, pooling33)
{
  // Test the assembly part only
  // Set to 17 to let 'int remain = outw - (nn << 2);' in the original code to be 0
  const int w = 17;
  const int h = 17;
  const int outw = w/2;
  const int outh = h/2;
  assert(outw==8);
  assert(outh==8);
  
  const int tailstep = w - 2*outw + w;
  
  int nn = outw >> 2;
  int nn2 = outw >> 2;
  
  float c0[w*h] __attribute__((aligned(16)));
  for (int i=0; i<w*h; i++)
    c0[i] = i+1;
  
  const float *r0 = c0;
  const float *r1 = c0 + w;
  const float *r2 = c0 + w + w;
  float *outptr = new float[outw*outh]();
  const float *r02 = c0;
  const float *r12 = c0 + w;
  const float *r22 = c0 + w + w;
  float *outptr2 = new float[outw*outh]();

  
// Line 48
// Intrinsic
        for (int i = 0; i < outh; i++)
        {
            nn2 = outw >> 2;
            float32x4x2_t _r0 = vld2q_f32(r02);
            float32x4x2_t _r1 = vld2q_f32(r12);
            float32x4x2_t _r2 = vld2q_f32(r22);
            for (; nn2>0; nn2--)
            {
                float32x4x2_t _r0n = vld2q_f32(r02+8);
                float32x4x2_t _r1n = vld2q_f32(r12+8);
                float32x4x2_t _r2n = vld2q_f32(r22+8);

                float32x4_t _max0 = vmaxq_f32(_r0.val[0], _r0.val[1]);
                float32x4_t _max1 = vmaxq_f32(_r1.val[0], _r1.val[1]);
                float32x4_t _max2 = vmaxq_f32(_r2.val[0], _r2.val[1]);

                float32x4_t _r02 = vextq_f32(_r0.val[0], _r0n.val[0], 1);
                float32x4_t _r12 = vextq_f32(_r1.val[0], _r1n.val[0], 1);
                float32x4_t _r22 = vextq_f32(_r2.val[0], _r2n.val[0], 1);

                _max0 = vmaxq_f32(_max0, _r02);
                _max1 = vmaxq_f32(_max1, _r12);
                _max2 = vmaxq_f32(_max2, _r22);

                float32x4_t _max = vmaxq_f32(vmaxq_f32(_max0, _max1), _max2);

                vst1q_f32(outptr2, _max);

                _r0 = _r0n;
                _r1 = _r1n;
                _r2 = _r2n;

                r02 += 8;
                r12 += 8;
                r22 += 8;
                outptr2 += 4;
            }
            
            r02 += tailstep;//1 + w;
            r12 += tailstep;//1 + w;
            r22 += tailstep;//1 + w;
        }
// Assembly 64

        for (int i = 0; i < outh; i++)
        {
            nn = outw >> 2;
            asm volatile(
                "prfm       pldl1keep, [%1, #256]       \n"
                "ld2        {v0.4s, v1.4s}, [%1], #32   \n"
                "prfm       pldl1keep, [%2, #256]       \n"
                "ld2        {v2.4s, v3.4s}, [%2], #32   \n"
                "prfm       pldl1keep, [%3, #256]       \n"
                "ld2        {v4.4s, v5.4s}, [%3], #32   \n"
                "0:                                     \n"

                "prfm       pldl1keep, [%1, #256]       \n"
                "ld2        {v6.4s, v7.4s}, [%1], #32   \n"

                "fmax       v12.4s, v0.4s, v1.4s        \n"
                "fmax       v13.4s, v2.4s, v3.4s        \n"

                "prfm       pldl1keep, [%2, #256]       \n"
                "ld2        {v8.4s, v9.4s}, [%2], #32   \n"

                "fmax       v14.4s, v4.4s, v5.4s        \n"
                "ext        v0.16b, v0.16b, v6.16b, #4  \n"

                "prfm       pldl1keep, [%3, #256]       \n"
                "ld2        {v10.4s, v11.4s}, [%3], #32 \n"

                "ext        v2.16b,  v2.16b, v8.16b, #4 \n"

                "fmax       v12.4s, v12.4s, v0.4s       \n"
                "ext        v4.16b, v4.16b, v10.16b, #4 \n"

                "fmax       v13.4s, v13.4s, v2.4s       \n"
                "fmax       v14.4s, v14.4s, v4.4s       \n"
                "fmax       v12.4s, v12.4s, v13.4s      \n"

                "orr        v0.16b, v6.16b, v6.16b      \n"
                "orr        v1.16b, v7.16b, v7.16b      \n"
                "fmax       v12.4s, v12.4s, v14.4s      \n"

                "orr        v2.16b, v8.16b, v8.16b      \n"
                "orr        v3.16b, v9.16b, v9.16b      \n"
                "orr        v4.16b, v10.16b, v10.16b    \n"
                "orr        v5.16b, v11.16b, v11.16b    \n"

                "subs       %w0, %w0, #1                \n"
                "st1        {v12.4s}, [%4], #16         \n"
                "bne        0b                          \n"
                "sub        %1, %1, #32                 \n"
                "sub        %2, %2, #32                 \n"
                "sub        %3, %3, #32                 \n"
                : "=r"(nn),     // %0
                  "=r"(r0),     // %1
                  "=r"(r1),     // %2
                  "=r"(r2),     // %3
                  "=r"(outptr)  // %4
                : "0"(nn),
                  "1"(r0),
                  "2"(r1),
                  "3"(r2),
                  "4"(outptr)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14"
            );

            r0 += tailstep;//1 + w;
            r1 += tailstep;//1 + w;
            r2 += tailstep;//1 + w;
        }

  outptr -= outw*outh;
  outptr2 -= outw*outh;
  
  for (int i=0; i<outw*outh; i++)
  {
    EXPECT_EQ(*(outptr+i), *(outptr2+i));
    EXPECT_EQ(*(outptr+i), 2*w+3+2*(i%outw)+(i/outw)*2*w);
  }



  delete []outptr;
  delete []outptr2;
  
}

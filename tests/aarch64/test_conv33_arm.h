#pragma once
#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>
#include "gtest/gtest.h"

#define ENABLE_BIAS 1

#ifdef ENABLE_BIAS
#include <algorithm>
#endif

TEST(aarch64_assembly, conv33)
{
  // *************************************************************
  // ********                                           **********
  // ********       Convolution 3*3 with stride 2       **********
  // ********                                           **********
  // *************************************************************
  
  
  // Test the assembly part only
  // Input is set to 10\11 respectively, so that outw=outh=9, let outh to be odd number.
  // Test is for 1 group only. (group is not considered)
  int w = 17;
  int h = 17;
  int outw = (w-1)/2;
  int outh = (h-1)/2;
  assert(outw==8);
  assert(outh==8);
  
  const int tailstep = w - 2*outw + w;
  
  int nn = outw >> 2;
  int nn2 = outw >> 2;
  
  float c0[w*h] __attribute__((aligned(16))) ;
  for (int i=0; i<w*h; i++)
  {
    c0[i] = i+1;
  }
  
  // Kernels, k0 for 
  float ker1[9] __attribute__((aligned(16)))  = {1,1,1,1,1,1,1,1,1}; // Test kernel for main part
  float* kernel0 = ker1;

  const float* k0 = kernel0;
  const float* k1 = kernel0 + 3;
  const float* k2 = kernel0 + 6; 
  
  float *r0 = c0;
  float *r1 = c0 + w;
  float *r2 = c0 + w * 2;
  float *r3 = c0 + w * 3;
  
  float *r0b = c0;
  float *r1b = c0 + w;
  float *r2b = c0 + w * 2;
  float *r3b = c0 + w * 3;
  float *out  = new float[outw*outh]();
  float *out2 = new float[outw*outh]();
  
  float* outptr = out;
  float* outptrb = out2;

#ifdef ENABLE_BIAS
  // Fill the bias
  const float bias0 = 10000.f;
    
  std::fill_n(out, outw*outh, bias0); 
  std::fill_n(out2, outw*outh, bias0); 
#endif

        // Shared part of both intrinsics and assembly
            float32x4_t _k0123 = vld1q_f32(k0);
            float32x4_t _k3456 = vld1q_f32(k1);
            float32x4_t _k6789 = vld1q_f32(k2);





// Line 6863
// Case: when stride = 2, compute 1*h at a time
// Intrinsic

            int i = 0;
            for (; i < outh; i++)
            {
                nn2 = outw >> 2;
                for (; nn2>0; nn2--)
                {
                    float32x4_t _outp = vld1q_f32(outptrb);

                    float32x4x2_t _r0 = vld2q_f32(r0b);
                    float32x4x2_t _r0n = vld2q_f32(r0b+8);

                    float32x4_t _r00 = _r0.val[0];// 0 2 4 6
                    float32x4_t _r01 = _r0.val[1];// 1 3 5 7
                    float32x4_t _r02 = vextq_f32(_r00, _r0n.val[0], 1);// 2 4 6 8

                    _outp = vfmaq_laneq_f32(_outp, _r00, _k0123, 0);
                    _outp = vfmaq_laneq_f32(_outp, _r01, _k0123, 1);
                    _outp = vfmaq_laneq_f32(_outp, _r02, _k0123, 2);

                    float32x4x2_t _r1 = vld2q_f32(r1b);
                    float32x4x2_t _r1n = vld2q_f32(r1b+8);

                    float32x4_t _r10 = _r1.val[0];
                    float32x4_t _r11 = _r1.val[1];
                    float32x4_t _r12 = vextq_f32(_r10, _r1n.val[0], 1);

                    _outp = vfmaq_laneq_f32(_outp, _r10, _k3456, 0);
                    _outp = vfmaq_laneq_f32(_outp, _r11, _k3456, 1);
                    _outp = vfmaq_laneq_f32(_outp, _r12, _k3456, 2);

                    float32x4x2_t _r2 = vld2q_f32(r2b);
                    float32x4x2_t _r2n = vld2q_f32(r2b+8);

                    float32x4_t _r20 = _r2.val[0];
                    float32x4_t _r21 = _r2.val[1];
                    float32x4_t _r22 = vextq_f32(_r20, _r2n.val[0], 1);

                    _outp = vfmaq_laneq_f32(_outp, _r20, _k6789, 0);
                    _outp = vfmaq_laneq_f32(_outp, _r21, _k6789, 1);
                    _outp = vfmaq_laneq_f32(_outp, _r22, _k6789, 2);

                    vst1q_f32(outptrb, _outp);

                    r0b += 8;
                    r1b += 8;
                    r2b += 8;
                    outptrb += 4;
                }

                r0b += tailstep;
                r1b += tailstep;
                r2b += tailstep;
            }


// Line 6908
// Case: when stride = 2, compute 1*h at a time
// Assembly

// Problem here: LD2 of aarch64 only supports { <Vt>.<T>, <Vt>.<T> }, {<Xn|SP>}
// So I use v8, plus v9 as a padding NEON register
// This unused operation (v9 padding) will affect the performance somehow. -.-||

            i = 0;
            for (; i < outh; i++)
            {
                nn = outw >> 2;

                if (nn > 0)
                {
                asm volatile(
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "0:                                        \n"
                    
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1]                  \n"

                    "fmla       v0.4s,  v2.4s, %10.s[0]        \n"
                    "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%2]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmul       v11.4s, v1.4s, %10.s[2]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                    "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                    "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%3]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %11.s[2]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                    "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                    "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%4]           \n"
                    "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                    "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                    "fadd       v0.4s, v0.4s, v10.4s           \n"
                    "fadd       v0.4s, v0.4s, v11.4s           \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s}, [%1], #16             \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #32                    \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2)      // %4
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "w"(_k0123),  // %10
                      "w"(_k3456),  // %11
                      "w"(_k6789)   // %12
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

  EXPECT_NE(outptr, outptrb);
  for (int i=0; i<outw*outh; i++)
  {
      EXPECT_NEAR(*(out+i), *(out2+i),1e-5);
//    printf("the %d of cc is : %f\n", i, *(out+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out2+i));
  }


  delete []out;
  delete []out2;

}

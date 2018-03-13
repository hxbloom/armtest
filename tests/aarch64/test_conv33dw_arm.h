#pragma once
#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>
#include "gtest/gtest.h"

TEST(aarch64_assembly, conv33dw)
{
  // Test the assembly part only
  // Input is set to 10\11 respectively, so that outw=outh=9, let outh to be odd number.
  // Test is for 1 group only. (group is not considered)
  int w = 10;
  int h = 11;
  int outw = w-2;
  int outh = h-2;
  assert(outw==8);
  assert(outh==9);
  
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
  
  const float bias0 = 10000.f;
  
  float *r0 = c0;
  float *r1 = c0 + w;
  float *r2 = c0 + w * 2;
  float *r3 = c0 + w * 3;
  
  float *r02 = c0;
  float *r12 = c0 + w;
  float *r22 = c0 + w * 2;
  float *r32 = c0 + w * 3;
  float *out  = new float[outw*outh]();
  float *out2 = new float[outw*outh]();
  
  float* outptr = out;
  float* outptr2 = outptr + outw;
  float* outptrb = out2;
  float* outptr2b = outptrb + outw;

        // Shared part of both intrinsics and assembly
        float32x4_t _k012x = vld1q_f32(kernel0);
        float32x4_t _k345x = vld1q_f32(kernel0+3);
        float32x4_t _k678x = vld1q_f32(kernel0+6);

        _k012x = vsetq_lane_f32(0.f, _k012x, 3);
        _k345x = vsetq_lane_f32(0.f, _k345x, 3);
        _k678x = vsetq_lane_f32(0.f, _k678x, 3);

        float32x4_t _bias0 = vdupq_n_f32(bias0);


// Line 80
// Case: when stride = 1, compute 2*h at a time
// Intrinsic

        int i = 0;

        for (; i+1 < outh; i+=2)
        {
            nn2 = outw >> 2;

            for (; nn2>0; nn2--)
            {
                float32x4_t _r00 = vld1q_f32(r02);
                float32x4_t _r00n = vld1q_f32(r02 + 4);
                float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);

                float32x4_t _r10 = vld1q_f32(r12);
                float32x4_t _r10n = vld1q_f32(r12 + 4);
                float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);

                float32x4_t _r20 = vld1q_f32(r22);
                float32x4_t _r20n = vld1q_f32(r22 + 4);
                float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);

                float32x4_t _r30 = vld1q_f32(r32);
                float32x4_t _r30n = vld1q_f32(r32 + 4);
                float32x4_t _r31 = vextq_f32(_r30, _r30n, 1);
                float32x4_t _r32 = vextq_f32(_r30, _r30n, 2);

                float32x4_t _sum1 = vmulq_laneq_f32(_r00, _k012x, 0);
                float32x4_t _sum2 = vfmaq_laneq_f32(_bias0, _r01, _k012x, 1);
                _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k012x, 2);
                _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k345x, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k345x, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k345x, 2);
                _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k678x, 0);
                _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k678x, 1);
                _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k678x, 2);

                float32x4_t _sum3 = vmulq_laneq_f32(_r10, _k012x, 0);
                float32x4_t _sum4 = vfmaq_laneq_f32(_bias0, _r11, _k012x, 1);
                _sum3 = vfmaq_laneq_f32(_sum3, _r12, _k012x, 2);
                _sum4 = vfmaq_laneq_f32(_sum4, _r20, _k345x, 0);
                _sum3 = vfmaq_laneq_f32(_sum3, _r21, _k345x, 1);
                _sum4 = vfmaq_laneq_f32(_sum4, _r22, _k345x, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _r30, _k678x, 0);
                _sum4 = vfmaq_laneq_f32(_sum4, _r31, _k678x, 1);
                _sum3 = vfmaq_laneq_f32(_sum3, _r32, _k678x, 2);

                _sum1 = vaddq_f32(_sum1, _sum2);
                _sum3 = vaddq_f32(_sum3, _sum4);

                vst1q_f32(outptrb, _sum1);
                vst1q_f32(outptr2b, _sum3);

                r02 += 4;
                r12 += 4;
                r22 += 4;
                r32 += 4;
                outptrb += 4;
                outptr2b += 4;
            }

            r02 += 2 + w;
            r12 += 2 + w;
            r22 += 2 + w;
            r32 += 2 + w;

            outptrb += outw;
            outptr2b += outw;
        }

// Case: when stride = 1, compute the last h (if exists) at a time
// Intrinsic
        for (; i < outh; i++)
        {
            nn2 = outw >> 2;

            for (; nn2>0; nn2--)
            {
                float32x4_t _r00 = vld1q_f32(r02);
                float32x4_t _r00n = vld1q_f32(r02 + 4);
                float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);

                float32x4_t _r10 = vld1q_f32(r12);
                float32x4_t _r10n = vld1q_f32(r12 + 4);
                float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);

                float32x4_t _r20 = vld1q_f32(r22);
                float32x4_t _r20n = vld1q_f32(r22 + 4);
                float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);

                float32x4_t _sum1 = vmulq_laneq_f32(_r00, _k012x, 0);
                float32x4_t _sum2 = vfmaq_laneq_f32(_bias0, _r01, _k012x, 1);
                _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k012x, 2);
                _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k345x, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k345x, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k345x, 2);
                _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k678x, 0);
                _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k678x, 1);
                _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k678x, 2);

                _sum1 = vaddq_f32(_sum1, _sum2);

                vst1q_f32(outptrb, _sum1);

                r02 += 4;
                r12 += 4;
                r22 += 4;
                outptrb += 4;
            }

            r02 += 2;
            r12 += 2;
            r22 += 2;
        }


// Case: when stride = 1, compute 2*h at a time
// Assembly
// Line 138
        i = 0;   
        for (; i+1 < outh; i+=2)
        {
            int nn = outw >> 2;

            if (nn > 0)
            {
            asm volatile(
                "prfm       pldl1keep, [%3, #192]          \n"
                "ld1        {v9.4s, v10.4s}, [%3]          \n" //r0
                "add        %3, %3, #16                    \n"

                "ext        v11.16b, v9.16b, v10.16b, #4   \n"
                "ext        v12.16b, v9.16b, v10.16b, #8   \n"

                "0:                                        \n"

                "fmul       v7.4s, v9.4s, %14.s[0]         \n"
                
                "and        v13.16b, %17.16b, %17.16b      \n" // v13 = _bias0
                "fmul       v6.4s,  v11.4s, %14.s[1]       \n"
                "fmla       v13.4s, v12.4s, %14.s[2]       \n"

                "prfm       pldl1keep, [%4, #192]          \n"
                "ld1        {v9.4s, v10.4s}, [%4]          \n"
                "add        %4, %4, #16                    \n"

                "fmla       v7.4s, v9.4s, %15.s[0]         \n"

                "ext        v11.16b, v9.16b, v10.16b, #4   \n"
                "ext        v12.16b, v9.16b, v10.16b, #8   \n"

                "fmla       v6.4s,  v11.4s, %15.s[1]       \n"
                "fmla       v13.4s, v12.4s, %15.s[2]       \n"

                "fmul       v8.4s, v9.4s, %14.s[0]         \n"

                "and        v15.16b, %17.16b, %17.16b      \n" // v15 = _bias0
                "fmul       v14.4s, v11.4s, %14.s[1]       \n"
                "fmla       v15.4s, v12.4s, %14.s[2]       \n"

                "prfm       pldl1keep, [%5, #192]          \n"
                "ld1        {v9.4s, v10.4s}, [%5]          \n"
                "add        %5, %5, #16                    \n"

                "fmla       v7.4s, v9.4s, %16.s[0]         \n"

                "ext        v11.16b, v9.16b, v10.16b, #4   \n"
                "ext        v12.16b, v9.16b, v10.16b, #8   \n"

                "fmla       v6.4s,  v11.4s, %16.s[1]       \n"
                "fmla       v13.4s, v12.4s, %16.s[2]       \n"

                "fmla       v8.4s,   v9.4s, %15.s[0]       \n"
                "fmla       v14.4s, v11.4s, %15.s[1]       \n"
                "fmla       v15.4s, v12.4s, %15.s[2]       \n"

                "prfm       pldl1keep, [%6, #192]          \n"
                "ld1        {v9.4s, v10.4s}, [%6]          \n"
                "add        %6, %6, #16                    \n"

                "fmla       v8.4s, v9.4s, %16.s[0]         \n"

                "ext        v11.16b, v9.16b, v10.16b, #4   \n"
                "ext        v12.16b, v9.16b, v10.16b, #8   \n"

                "fmla       v14.4s, v11.4s, %16.s[1]       \n"
                "fmla       v15.4s, v12.4s, %16.s[2]       \n"

                "fadd       v7.4s, v7.4s, v6.4s            \n"

                "prfm       pldl1keep, [%3, #192]          \n"
                "ld1        {v9.4s, v10.4s}, [%3]          \n" //ro, for next loop

                "fadd       v8.4s, v8.4s, v14.4s           \n"
                "fadd       v7.4s, v7.4s, v13.4s           \n"
                "fadd       v8.4s, v8.4s, v15.4s           \n"

                "ext        v11.16b, v9.16b, v10.16b, #4   \n" // for next loop
                "ext        v12.16b, v9.16b, v10.16b, #8   \n" // for next loop

                "add        %3, %3, #16                    \n"
                
                "st1        {v7.4s}, [%1], #16             \n"
                "st1        {v8.4s}, [%2], #16             \n"

                "subs       %w0, %w0, #1                   \n"
                "bne        0b                             \n"

                "sub        %3, %3, #16                    \n"
                : "=r"(nn),         // %0
                  "=r"(outptr),     // %1
                  "=r"(outptr2),    // %2
                  "=r"(r0),         // %3
                  "=r"(r1),         // %4
                  "=r"(r2),         // %5
                  "=r"(r3)          // %6
                : "0"(nn),
                  "1"(outptr),
                  "2"(outptr2),
                  "3"(r0),
                  "4"(r1),
                  "5"(r2),
                  "6"(r3),
                  "w"(_k012x),      // %14
                  "w"(_k345x),      // %15
                  "w"(_k678x),      // %16
                  "w"(_bias0)       // %17
                : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            );
            }

            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr += outw;
            outptr2 += outw;
        }

// Case: when stride = 1, compute the last h (if exists) at a time
// Assembly
// Line 368
        for (; i < outh; i++)
        {
            nn = outw >> 2;

            if (nn > 0)
            {
            asm volatile(
                "prfm       pldl1keep, [%2, #192]          \n"
                "ld1        {v8.4s, v9.4s}, [%2]           \n" //r0
                "add        %2, %2, #16                    \n"

                "ext        v10.16b, v8.16b, v9.16b, #4    \n"
                "ext        v11.16b, v8.16b, v9.16b, #8    \n"

                "0:                                        \n"

                "fmul       v7.4s, v8.4s, %10.s[0]         \n"

                "and        v14.16b, %13.16b, %13.16b      \n" // v14 = _bias0
                "fmul       v13.4s, v10.4s, %10.s[1]       \n"
                "fmla       v14.4s, v11.4s, %10.s[2]       \n"

                "prfm       pldl1keep, [%3, #192]          \n"
                "ld1        {v8.4s, v9.4s}, [%3]           \n" //r1
                "add        %3, %3, #16                    \n"

                "fmla       v7.4s, v8.4s, %11.s[0]         \n"

                "ext        v10.16b, v8.16b, v9.16b, #4    \n"
                "ext        v11.16b, v8.16b, v9.16b, #8    \n"

                "fmla       v13.4s, v10.4s, %11.s[1]       \n"
                "fmla       v14.4s, v11.4s, %11.s[2]       \n"

                "prfm       pldl1keep, [%4, #192]          \n"
                "ld1        {v8.4s, v9.4s}, [%4]           \n" //r2
                "add        %4, %4, #16                    \n"

                "fmla       v7.4s, v8.4s, %12.s[0]       \n"

                "ext        v10.16b, v8.16b, v9.16b, #4    \n"
                "ext        v11.16b, v8.16b, v9.16b, #8    \n"

                "fmla       v13.4s, v10.4s, %12.s[1]       \n"
                "fmla       v14.4s, v11.4s, %12.s[2]       \n"
                
                "prfm       pldl1keep, [%2, #192]          \n"
                "ld1        {v8.4s, v9.4s}, [%2]           \n" //r0, for next loop
                "add        %2, %2, #16                    \n"

                "fadd       v7.4s, v7.4s, v13.4s           \n"
                "fadd       v7.4s, v7.4s, v14.4s           \n"

                "ext        v10.16b, v8.16b, v9.16b, #4    \n"  // for next loop
                "ext        v11.16b, v8.16b, v9.16b, #8    \n"  // for next loop

                "st1        {v7.4s}, [%1], #16             \n"

                "subs       %w0, %w0, #1                   \n"
                "bne        0b                             \n"

                "sub        %2, %2, #16                    \n"
                : "=r"(nn),         // %0
                  "=r"(outptr),     // %1
                  "=r"(r0),         // %2
                  "=r"(r1),         // %3
                  "=r"(r2)          // %4
                : "0"(nn),
                  "1"(outptr),
                  "2"(r0),
                  "3"(r1),
                  "4"(r2),
                  "w"(_k012x),      // %10
                  "w"(_k345x),      // %11
                  "w"(_k678x),      // %12
                  "w"(_bias0)       // %13
                : "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            );
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }

  outptr -= outw*outh;
  outptrb -= outw*outh;
  
  EXPECT_NE(outptr, outptrb);
  for (int i=0; i<outw*outh; i++)
  {
      EXPECT_EQ(*(out+i), *(out2+i));
//    printf("the %d of cc is : %f\n", i, *(out+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out2+i));
  }
    
// Line 522
// Case: when stride = 2, compute 1*h at a time
// Input is set to 10\11 respectively, so that outw=outh=9, let outh to be odd number.
// Test is for 1 group only. (group is not considered)

  w = 17;
  h = 17;
  outw = (w-1)/2;
  outh = (w-1)/2;
  assert(outw==8);
  assert(outh==8);
  
  const int tailstep = w - 2*outw + w;
  
  nn = outw >> 2;
  nn2 = outw >> 2;
  
  float d0[w*h] __attribute__((aligned(16))) ;
  for (int i=0; i<w*h; i++)
  {
    d0[i] = i+1;
  }

  r0 = d0;
  r1 = d0 + w;
  r2 = d0 + w * 2;
  r3 = d0 + w * 3;
  
  r02 = d0;
  r12 = d0 + w;
  r22 = d0 + w * 2;
  r32 = d0 + w * 3;
  float *out3  = new float[outw*outh]();
  float *out4  = new float[outw*outh]();
  
  outptr = out3;
  outptrb = out4;

  float ker2[9] __attribute__((aligned(16)))  = {1,1,1,1,1,1,1,1,1}; // Test kernel for remaining
  kernel0 = ker2;
  
        _k012x = vld1q_f32(kernel0);
        _k345x = vld1q_f32(kernel0+3);
        _k678x = vld1q_f32(kernel0+6);

        _k012x = vsetq_lane_f32(0.f, _k012x, 3);
        _k345x = vsetq_lane_f32(0.f, _k345x, 3);
        _k678x = vsetq_lane_f32(0.f, _k678x, 3);

        _bias0 = vdupq_n_f32(bias0);

// Line 550
// Case: when stride = 2, compute 1*h at a time
// Intrinsic

        i = 0;
        for (; i < outh; i++)
        {
            nn2 = outw >> 2;

            for (; nn2>0; nn2--)
            {
                float32x4x2_t _r0 = vld2q_f32(r02);
                float32x4x2_t _r0n = vld2q_f32(r02+8);

                float32x4_t _r00 = _r0.val[0];// 0 2 4 6
                float32x4_t _r01 = _r0.val[1];// 1 3 5 7
                float32x4_t _r02 = vextq_f32(_r00, _r0n.val[0], 1);// 2 4 6 8

                float32x4_t _outp = vfmaq_laneq_f32(_bias0, _r00, _k012x, 0);
                _outp = vfmaq_laneq_f32(_outp, _r01, _k012x, 1);
                _outp = vfmaq_laneq_f32(_outp, _r02, _k012x, 2);

                float32x4x2_t _r1 = vld2q_f32(r12);
                float32x4x2_t _r1n = vld2q_f32(r12+8);

                float32x4_t _r10 = _r1.val[0];
                float32x4_t _r11 = _r1.val[1];
                float32x4_t _r12 = vextq_f32(_r10, _r1n.val[0], 1);

                _outp = vfmaq_laneq_f32(_outp, _r10, _k345x, 0);
                _outp = vfmaq_laneq_f32(_outp, _r11, _k345x, 1);
                _outp = vfmaq_laneq_f32(_outp, _r12, _k345x, 2);

                float32x4x2_t _r2 = vld2q_f32(r22);
                float32x4x2_t _r2n = vld2q_f32(r22+8);

                float32x4_t _r20 = _r2.val[0];
                float32x4_t _r21 = _r2.val[1];
                float32x4_t _r22 = vextq_f32(_r20, _r2n.val[0], 1);

                _outp = vfmaq_laneq_f32(_outp, _r20, _k678x, 0);
                _outp = vfmaq_laneq_f32(_outp, _r21, _k678x, 1);
                _outp = vfmaq_laneq_f32(_outp, _r22, _k678x, 2);

                vst1q_f32(outptrb, _outp);

                r02 += 8;
                r12 += 8;
                r22 += 8;
                outptrb += 4;
            }


            r02 += tailstep;
            r12 += tailstep;
            r22 += tailstep;
        }


// Line 595
// Case: when stride = 2, compute 1*h at a time
// Assembly

        i = 0;
        for (; i < outh; i++)
        {
            nn = outw >> 2;

// Problem here: LD2 of aarch64 only supports { <Vt>.<T>, <Vt>.<T> }, {<Xn|SP>}
// So I replace v8 with v19, then add v20 here as a padding NEON register
// This unused operation will affect the performance somehow. -.-||

            if (nn > 0)
            {
            asm volatile(
                "prfm       pldl1keep, [%2, #256]          \n"
                "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                "and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

                "0:                                        \n"
                "fmul       v0.4s,  v2.4s, %10.s[0]        \n"
                "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                "prfm       pldl1keep, [%2, #256]          \n"
                "ld2        {v19.4s, v20.4s}, [%2]         \n"
                "ext        v1.16b, v2.16b, v19.16b, #4    \n"

                "fmla       v11.4s, v1.4s, %10.s[2]        \n"

                "prfm       pldl1keep, [%3, #256]          \n"
                "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                "prfm       pldl1keep, [%3, #256]          \n"
                "ld2        {v19.4s, v20.4s}, [%3]         \n"
                "ext        v1.16b, v2.16b, v19.16b, #4    \n"

                "fmla       v11.4s, v1.4s, %11.s[2]        \n"
                
                "prfm       pldl1keep, [%4, #256]          \n"
                "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                "prfm       pldl1keep, [%4, #256]          \n"
                "ld2        {v19.4s, v20.4s}, [%4]         \n"
                "ext        v1.16b, v2.16b, v19.16b, #4    \n"

                "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                "prfm       pldl1keep, [%2, #256]          \n"
                "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                "fadd       v0.4s, v0.4s, v10.4s           \n"
                "fadd       v0.4s, v0.4s, v11.4s           \n"

                "and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

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
                  "w"(_k012x),  // %10
                  "w"(_k345x),  // %11
                  "w"(_k678x),  // %12
                  "w"(_bias0)   // %13
                : "cc", "memory", "v0", "v1", "v2", "v3", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v19", "v20"
            );
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }


  outptr -= outw*outh;
  outptrb -= outw*outh;
  
  EXPECT_NE(outptr, outptrb);
  for (int i=0; i<outw*outh; i++)
  {
      EXPECT_EQ(*(out3+i), *(out4+i));
//    printf("the %d of cc is : %f\n", i, *(out3+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out4+i));
  }


  delete []out;
  delete []out2;
  delete []out3;
  delete []out4;
    
}

#pragma once
#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>
#include "gtest/gtest.h"


#define ENABLE_BIAS 1

#ifdef ENABLE_BIAS
#include <algorithm>
#endif

TEST(aarch64_assembly, conv77)
{
  // *************************************************************
  // ********                                           **********
  // ********   Part 1: Convolution 7*7 with stride 1   **********
  // ********                                           **********
  // *************************************************************
  
  
  // Test the assembly part only
  // Input is set to 4/4 respectively, so that outw=outh=8, let outh to be odd number.
  // Test is for 1 input channel only. (multiply input channel case is not considered)
  // The function of accumulation of from different input channels can be tests by adding the bias
  // To check if the code is indeed accululating the results, not start from 0.f every time.
  int w = 14;
  int h = 14;
  int outw = w-6;
  int outh = h-6;
  assert(outw==8);
  assert(outh==8);
  
  int nn = outw >> 2;
  int nn2 = outw >> 2;
  
  float c0[w*h] __attribute__((aligned(16))) ;
  for (int i=0; i<w*h; i++)
  {
    c0[i] = i+1;
  }
  
  // Kernels, k0 for 
  float ker1[49] __attribute__((aligned(16)))  = {2,2,2,2,2,2,2,    1,1,7,7,1,1,1, \
                                                  1,1,1,1,2,2,1,    3,3,3,3,1,1,1, \
                                                  1,1,4,4,4,1,1,    1,8,1,1,1,8,1, \
                                                  1,1,1,1,6,6,1                    }; // Test kernel for main part
  float* kernel0 = ker1;

  float* k0 = kernel0;
  float* k1 = kernel0 + 7;
  float* k2 = kernel0 + 14; 
  float* k3 = kernel0 + 21;
  float* k4 = kernel0 + 28;
  float* k5 = kernel0 + 35;
  float* k6 = kernel0 + 42;
  
  
  float *r0 = c0;
  float *r1 = c0 + w;
  float *r2 = c0 + w * 2;
  float *r3 = c0 + w * 3;
  float *r4 = c0 + w * 4;
  float *r5 = c0 + w * 5;
  float *r6 = c0 + w * 6;
  
  float *r0b = c0;
  float *r1b = c0 + w;
  float *r2b = c0 + w * 2;
  float *r3b = c0 + w * 3;
  float *r4b = c0 + w * 4;
  float *r5b = c0 + w * 5;
  float *r6b = c0 + w * 6;
  
  float *out  = new float[outw*outh]();
  float *out2 = new float[outw*outh]();
  
  float* outptr = out;
  float* outptrb = out2;

#ifdef ENABLE_BIAS
  // Fill the bias
  float bias0 = 10000.f;
    
  std::fill_n(out, outw*outh, bias0); 
  std::fill_n(out2, outw*outh, bias0); 
#endif


#if __ARM_NEON && __aarch64__
                    float32x4_t _k0123 = vld1q_f32(k0);
                    float32x4_t _k4567 = vld1q_f32(k0 + 4);
                    float32x4_t _k78910 = vld1q_f32(k1);
                    float32x4_t _k11121314 = vld1q_f32(k1 + 4);
                    float32x4_t _k14151617 = vld1q_f32(k2);
                    float32x4_t _k18192021 = vld1q_f32(k2 + 4);
                    float32x4_t _k21222324 = vld1q_f32(k3);
                    float32x4_t _k25262728 = vld1q_f32(k3 + 4);
                    float32x4_t _k28293031 = vld1q_f32(k4);
                    float32x4_t _k32333435 = vld1q_f32(k4 + 4);
                    float32x4_t _k35363738 = vld1q_f32(k5);
                    float32x4_t _k39404142 = vld1q_f32(k5 + 4);
                    float32x4_t _k42434445 = vld1q_f32(k6);
                    float32x4_t _k46474849 = vld1q_f32(k6 + 4);
#endif // __ARM_NEON && __aarch64__



#if __ARM_NEON && __aarch64__
#ifdef __clang__    // __ARM_NEON && __aarch64__ && __clang__


#else   // __ARM_NEON && __aarch64__ defined, but __clang__ not defined


#endif   // __clang__


#endif // __ARM_NEON && __aarch64__





// Line 78
// Intrinsic

#if __aarch64__
            int i = 0;
            for (; i < outh; i++)
            {
                nn2 = outw >> 2;

                for (; nn2>0; nn2--)
                {
                    float32x4_t _sum = vld1q_f32(outptrb);

                    float32x4_t _r00 = vld1q_f32(r0b);// 0 1 2 3
                    float32x4_t _r04 = vld1q_f32(r0b + 4);// 4 5 6 7
                    float32x4_t _r00n = vld1q_f32(r0b + 8);// 8 9 10 11
                    float32x4_t _r01 = vextq_f32(_r00, _r04, 1);// 1 2 3 4
                    float32x4_t _r02 = vextq_f32(_r00, _r04, 2);// 2 3 4 5
                    float32x4_t _r03 = vextq_f32(_r00, _r04, 3);// 3 4 5 6
                    float32x4_t _r05 = vextq_f32(_r04, _r00n, 1);// 5 6 7 8
                    float32x4_t _r06 = vextq_f32(_r04, _r00n, 2);// 6 7 8 9

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r05, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r06, _k4567, 2);

                    float32x4_t _r10 = vld1q_f32(r1b);
                    float32x4_t _r14 = vld1q_f32(r1b + 4);
                    float32x4_t _r10n = vld1q_f32(r1b + 8);
                    float32x4_t _r11 = vextq_f32(_r10, _r14, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r14, 2);
                    float32x4_t _r13 = vextq_f32(_r10, _r14, 3);
                    float32x4_t _r15 = vextq_f32(_r14, _r10n, 1);
                    float32x4_t _r16 = vextq_f32(_r14, _r10n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k78910, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k78910, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k78910, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k78910, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k11121314, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r15, _k11121314, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r16, _k11121314, 2);


                    float32x4_t _r20 = vld1q_f32(r2b);
                    float32x4_t _r24 = vld1q_f32(r2b + 4);
                    float32x4_t _r20n = vld1q_f32(r2b + 8);
                    float32x4_t _r21 = vextq_f32(_r20, _r24, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r24, 2);
                    float32x4_t _r23 = vextq_f32(_r20, _r24, 3);
                    float32x4_t _r25 = vextq_f32(_r24, _r20n, 1);
                    float32x4_t _r26 = vextq_f32(_r24, _r20n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k14151617, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k14151617, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k14151617, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k14151617, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k18192021, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r25, _k18192021, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r26, _k18192021, 2);

                    float32x4_t _r30 = vld1q_f32(r3b);
                    float32x4_t _r34 = vld1q_f32(r3b + 4);
                    float32x4_t _r30n = vld1q_f32(r3b + 8);
                    float32x4_t _r31 = vextq_f32(_r30, _r34, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r34, 2);
                    float32x4_t _r33 = vextq_f32(_r30, _r34, 3);
                    float32x4_t _r35 = vextq_f32(_r34, _r30n, 1);
                    float32x4_t _r36 = vextq_f32(_r34, _r30n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k21222324, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k21222324, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k21222324, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k21222324, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k25262728, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r35, _k25262728, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r36, _k25262728, 2);

                    float32x4_t _r40 = vld1q_f32(r4b);
                    float32x4_t _r44 = vld1q_f32(r4b + 4);
                    float32x4_t _r40n = vld1q_f32(r4b + 8);
                    float32x4_t _r41 = vextq_f32(_r40, _r44, 1);
                    float32x4_t _r42 = vextq_f32(_r40, _r44, 2);
                    float32x4_t _r43 = vextq_f32(_r40, _r44, 3);
                    float32x4_t _r45 = vextq_f32(_r44, _r40n, 1);
                    float32x4_t _r46 = vextq_f32(_r44, _r40n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k28293031, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k28293031, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k28293031, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k28293031, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k32333435, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r45, _k32333435, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r46, _k32333435, 2);


                    float32x4_t _r50 = vld1q_f32(r5b);
                    float32x4_t _r54 = vld1q_f32(r5b + 4);
                    float32x4_t _r50n = vld1q_f32(r5b + 8);
                    float32x4_t _r51 = vextq_f32(_r50, _r54, 1);
                    float32x4_t _r52 = vextq_f32(_r50, _r54, 2);
                    float32x4_t _r53 = vextq_f32(_r50, _r54, 3);
                    float32x4_t _r55 = vextq_f32(_r54, _r50n, 1);
                    float32x4_t _r56 = vextq_f32(_r54, _r50n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r50, _k35363738, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r51, _k35363738, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r52, _k35363738, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r53, _k35363738, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r54, _k39404142, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r55, _k39404142, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r56, _k39404142, 2);

                    float32x4_t _r60 = vld1q_f32(r6b);
                    float32x4_t _r64 = vld1q_f32(r6b + 4);
                    float32x4_t _r60n = vld1q_f32(r6b + 8);
                    float32x4_t _r61 = vextq_f32(_r60, _r64, 1);
                    float32x4_t _r62 = vextq_f32(_r60, _r64, 2);
                    float32x4_t _r63 = vextq_f32(_r60, _r64, 3);
                    float32x4_t _r65 = vextq_f32(_r64, _r60n, 1);
                    float32x4_t _r66 = vextq_f32(_r64, _r60n, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r60, _k42434445, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r61, _k42434445, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r62, _k42434445, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r63, _k42434445, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r64, _k46474849, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r65, _k46474849, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r66, _k46474849, 2);

                    vst1q_f32(outptrb, _sum);

                    r0b += 4;
                    r1b += 4;
                    r2b += 4;
                    r3b += 4;
                    r4b += 4;
                    r5b += 4;
                    r6b += 4;
                    outptrb += 4;
                }

                r0b += 6;
                r1b += 6;
                r2b += 6;
                r3b += 6;
                r4b += 6;
                r5b += 6;
                r6b += 6;
            }
#endif //__aarch64__

// Line 236
// Assembly

// ************************** more than 30 operands used ***********************************
// ************************** place under #if __clang__  ***********************************

            i = 0;
            for (; i < outh; i++)
            {
                nn = outw >> 2;

                if (nn > 0)
                {
                asm volatile(
                    // v0:  input / final output
                    // v1 v2 v3: = ri0 ri4 ri0n , i <-  1-7
                    // v4 = ri1 / ri3 / ri6
                    // v5 = ri2 / ri5
                    // v9 = intermediate sum register
                    "0:                                        \n"                    
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1]                  \n"

                    //i = 1
                    "prfm       pldl1keep, [%2, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%2]    \n"
                    "add        %2, %2, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmul       v9.4s, v1.4s, %18.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %18.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %18.s[2]         \n"
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %18.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %19.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %19.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %19.s[2]         \n"

                    //i = 2
                    "prfm       pldl1keep, [%3, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%3]    \n" // v1 v2 v3: = r20 r24 r20n
                    "add        %3, %3, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n" // v4 = r21
                    "fmla       v9.4s, v1.4s, %20.s[0]         \n" // *+ r10
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n" // v5 = r22
                    "fmla       v0.4s, v4.4s, %20.s[1]         \n" // *+ r11
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n" // v4 = r23
                    "fmla       v9.4s, v5.4s, %20.s[2]         \n" // *+ r1
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n" // v5 = r25
                    "fmla       v0.4s, v4.4s, %20.s[3]         \n" // *+ r13
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n" // v4 = r26
                    "fmla       v9.4s, v2.4s, %21.s[0]         \n" // *+ r14
                    "fmla       v0.4s, v5.4s, %21.s[1]         \n" // *+ r15
                    "fmla       v9.4s, v4.4s, %21.s[2]         \n" // *+ r16

                    //i = 3
                    "prfm       pldl1keep, [%4, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%4]    \n"
                    "add        %4, %4, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmla       v9.4s, v1.4s, %22.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %22.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %22.s[2]         \n"
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %22.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %23.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %23.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %23.s[2]         \n"

                    //i = 4
                    "prfm       pldl1keep, [%5, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%5]    \n"
                    "add        %5, %5, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmla       v9.4s, v1.4s, %24.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %24.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %24.s[2]         \n"
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %24.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %25.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %25.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %25.s[2]         \n"

                    //i = 5
                    "prfm       pldl1keep, [%6, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%6]    \n"
                    "add        %6, %6, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmla       v9.4s, v1.4s, %26.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %26.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %26.s[2]         \n"
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %26.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %27.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %27.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %27.s[2]         \n"

                    //i = 6
                    "prfm       pldl1keep, [%7, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%7]    \n"
                    "add        %7, %7, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmla       v9.4s, v1.4s, %28.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %28.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %28.s[2]         \n"
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %28.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %29.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %29.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %29.s[2]         \n"
                    
                    //i = 7
                    "prfm       pldl1keep, [%8, #384]          \n"
                    "ld1        {v1.4s, v2.4s, v3.4s}, [%8]    \n"
                    "add        %8, %8, #16                    \n"
                    "ext        v4.16b, v1.16b, v2.16b, #4     \n"
                    "fmla       v9.4s, v1.4s, %30.s[0]         \n"
                    "ext        v5.16b, v1.16b, v2.16b, #8     \n"
                    "fmla       v0.4s, v4.4s, %30.s[1]         \n"
                    "ext        v4.16b, v1.16b, v2.16b, #12    \n"
                    "fmla       v9.4s, v5.4s, %30.s[2]         \n"                    
                    "ext        v5.16b, v2.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v4.4s, %30.s[3]         \n"
                    "ext        v4.16b, v2.16b, v3.16b, #8     \n"
                    "fmla       v9.4s, v2.4s, %31.s[0]         \n"
                    "fmla       v0.4s, v5.4s, %31.s[1]         \n"
                    "fmla       v9.4s, v4.4s, %31.s[2]         \n"

                    "fadd       v0.4s, v0.4s, v9.4s            \n"                    
                    "st1        {v0.4s}, [%1], #16             \n"                    
                    "subs       %w0, %w0, #1                   \n"
                    "bne        0b                             \n"
                    
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4),         // %6
                      "=r"(r5),         // %7
                      "=r"(r6)          // %8
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(r5),
                      "8"(r6),
                      "w"(_k0123),     // %18
                      "w"(_k4567),     // %19
                      "w"(_k78910),    // %20
                      "w"(_k11121314), // %21
                      "w"(_k14151617), // %22
                      "w"(_k18192021), // %23
                      "w"(_k21222324), // %24
                      "w"(_k25262728), // %25
                      "w"(_k28293031), // %26
                      "w"(_k32333435), // %27
                      "w"(_k35363738), // %28
                      "w"(_k39404142), // %29
                      "w"(_k42434445), // %30
                      "w"(_k46474849)  // %31
                    : "cc", "memory","v0", "v1", "v2", "v3", "v4", "v5", "v9"
                );
                }                    

                r0 += 6;
                r1 += 6;
                r2 += 6;
                r3 += 6;
                r4 += 6;
                r5 += 6;
                r6 += 6;
            }

  EXPECT_NE(outptr, outptrb);
  for (int i=0; i<outw*outh; i++)
  {
      EXPECT_NEAR(*(out+i), *(out2+i),1e-5);
//    printf("the %d of cc is : %f\n", i, *(out+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out2+i));
  }

















  // *************************************************************
  // ********                                           **********
  // ********   Part 2: Convolution 7*7 with stride 2   **********
  // ********                                           **********
  // *************************************************************


  w = 21;
  h = 21;
  outw = (w-5)/2;
  outh = (w-5)/2;
  assert(outw==8);
  assert(outh==8);

  nn = outw >> 2;
  nn2 = outw >> 2;
  
  const int tailstep = w - 2*outw + w;
  
  float d0[w*h] __attribute__((aligned(16))) ;
  for (int i=0; i<w*h; i++)
  {
    d0[i] = i+1;
  }
  
  // Kernels, k0 for 
  float ker2[49] __attribute__((aligned(16)))  = {1,6,6,1,1,1,1,    9,9,1,1,1,1,9, \
                                                  1,1,4,4,4,1,1,    2,2,2,2,2,2,2, \
                                                  1,1,1,1,5,5,5,    1,1,1,1,8,8,1, \
                                                  3,3,3,1,1,1,1                    }; // Test kernel for main part
  kernel0 = ker2;

  k0 = kernel0;
  k1 = kernel0 + 7;
  k2 = kernel0 + 14; 
  k3 = kernel0 + 21;
  k4 = kernel0 + 28;
  k5 = kernel0 + 35;
  k6 = kernel0 + 42;
  
  
  r0 = d0;
  r1 = d0 + w;
  r2 = d0 + w * 2;
  r3 = d0 + w * 3;
  r4 = d0 + w * 4;
  r5 = d0 + w * 5;
  r6 = d0 + w * 6;
  
  r0b = d0;
  r1b = d0 + w;
  r2b = d0 + w * 2;
  r3b = d0 + w * 3;
  r4b = d0 + w * 4;
  r5b = d0 + w * 5;
  r6b = d0 + w * 6;
  
  float *out3 = new float[outw*outh]();
  float *out4 = new float[outw*outh]();
  
  outptr = out3;
  outptrb = out4;

#ifdef ENABLE_BIAS
  // Fill the bias
  bias0 = 100000.f;
    
  std::fill_n(out3, outw*outh, bias0); 
  std::fill_n(out4, outw*outh, bias0); 
#endif


#if __ARM_NEON && __aarch64__
                    _k0123 = vld1q_f32(k0);
                    _k4567 = vld1q_f32(k0 + 4);
                    _k78910 = vld1q_f32(k1);
                    _k11121314 = vld1q_f32(k1 + 4);
                    _k14151617 = vld1q_f32(k2);
                    _k18192021 = vld1q_f32(k2 + 4);
                    _k21222324 = vld1q_f32(k3);
                    _k25262728 = vld1q_f32(k3 + 4);
                    _k28293031 = vld1q_f32(k4);
                    _k32333435 = vld1q_f32(k4 + 4);
                    _k35363738 = vld1q_f32(k5);
                    _k39404142 = vld1q_f32(k5 + 4);
                    _k42434445 = vld1q_f32(k6);
                    _k46474849 = vld1q_f32(k6 + 4);
#endif // __ARM_NEON && __aarch64__






// Line 602
// Intrinsic
            i = 0;
            for (; i < outh; i++)
            {
                nn2 = outw >> 2;

                for (; nn2>0; nn2--)
                {
                    float32x4_t _sum = vld1q_f32(outptrb);

                    float32x4x2_t _r00_02461357 = vld2q_f32(r0b);
                    float32x4x2_t _r00nx2 = vld2q_f32(r0b + 8);
                    float32x4_t _r0_8101214 = _r00nx2.val[0];// 8 10 12 14
                    float32x4_t _r0_9111315 = _r00nx2.val[1];// 9 11 13 15
                    float32x4_t _r00 = _r00_02461357.val[0];// 0 2 4 6
                    float32x4_t _r01 = _r00_02461357.val[1];// 1 3 5 7
                    float32x4_t _r02 = vextq_f32(_r00, _r0_8101214, 1);// 2 4 6 8
                    float32x4_t _r03 = vextq_f32(_r01, _r0_9111315, 1);// 3 5 7 9
                    float32x4_t _r04 = vextq_f32(_r00, _r0_8101214, 2);// 4 6 8 10
                    float32x4_t _r05 = vextq_f32(_r01, _r0_9111315, 2);// 5 7 9 11
                    float32x4_t _r06 = vextq_f32(_r00, _r0_8101214, 3);// 6 8 10 12

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r05, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r06, _k4567, 2);

                    float32x4x2_t _r10_02461357 = vld2q_f32(r1b);
                    float32x4x2_t _r10nx2 = vld2q_f32(r1b + 8);
                    float32x4_t _r1_8101214 = _r10nx2.val[0];
                    float32x4_t _r1_9111315 = _r10nx2.val[1];
                    float32x4_t _r10 = _r10_02461357.val[0];
                    float32x4_t _r11 = _r10_02461357.val[1];
                    float32x4_t _r12 = vextq_f32(_r10, _r1_8101214, 1);
                    float32x4_t _r13 = vextq_f32(_r11, _r1_9111315, 1);
                    float32x4_t _r14 = vextq_f32(_r10, _r1_8101214, 2);
                    float32x4_t _r15 = vextq_f32(_r11, _r1_9111315, 2);
                    float32x4_t _r16 = vextq_f32(_r10, _r1_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k78910, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k78910, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k78910, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k78910, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k11121314, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r15, _k11121314, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r16, _k11121314, 2);

                    float32x4x2_t _r20_02461357 = vld2q_f32(r2b);
                    float32x4x2_t _r20nx2 = vld2q_f32(r2b + 8);
                    float32x4_t _r2_8101214 = _r20nx2.val[0];
                    float32x4_t _r2_9111315 = _r20nx2.val[1];
                    float32x4_t _r20 = _r20_02461357.val[0];
                    float32x4_t _r21 = _r20_02461357.val[1];
                    float32x4_t _r22 = vextq_f32(_r20, _r2_8101214, 1);
                    float32x4_t _r23 = vextq_f32(_r21, _r2_9111315, 1);
                    float32x4_t _r24 = vextq_f32(_r20, _r2_8101214, 2);
                    float32x4_t _r25 = vextq_f32(_r21, _r2_9111315, 2);
                    float32x4_t _r26 = vextq_f32(_r20, _r2_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k14151617, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k14151617, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k14151617, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k14151617, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k18192021, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r25, _k18192021, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r26, _k18192021, 2);

                    float32x4x2_t _r30_02461357 = vld2q_f32(r3b);
                    float32x4x2_t _r30nx2 = vld2q_f32(r3b + 8);
                    float32x4_t _r3_8101214 = _r30nx2.val[0];
                    float32x4_t _r3_9111315 = _r30nx2.val[1];
                    float32x4_t _r30 = _r30_02461357.val[0];
                    float32x4_t _r31 = _r30_02461357.val[1];
                    float32x4_t _r32 = vextq_f32(_r30, _r3_8101214, 1);
                    float32x4_t _r33 = vextq_f32(_r31, _r3_9111315, 1);
                    float32x4_t _r34 = vextq_f32(_r30, _r3_8101214, 2);
                    float32x4_t _r35 = vextq_f32(_r31, _r3_9111315, 2);
                    float32x4_t _r36 = vextq_f32(_r30, _r3_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k21222324, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k21222324, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k21222324, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k21222324, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k25262728, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r35, _k25262728, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r36, _k25262728, 2);


                    float32x4x2_t _r40_02461357 = vld2q_f32(r4b);
                    float32x4x2_t _r40nx2 = vld2q_f32(r4b + 8);
                    float32x4_t _r4_8101214 = _r40nx2.val[0];
                    float32x4_t _r4_9111315 = _r40nx2.val[1];
                    float32x4_t _r40 = _r40_02461357.val[0];
                    float32x4_t _r41 = _r40_02461357.val[1];
                    float32x4_t _r42 = vextq_f32(_r40, _r4_8101214, 1);
                    float32x4_t _r43 = vextq_f32(_r41, _r4_9111315, 1);
                    float32x4_t _r44 = vextq_f32(_r40, _r4_8101214, 2);
                    float32x4_t _r45 = vextq_f32(_r41, _r4_9111315, 2);
                    float32x4_t _r46 = vextq_f32(_r40, _r4_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k28293031, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k28293031, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k28293031, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k28293031, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k32333435, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r45, _k32333435, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r46, _k32333435, 2);


                    float32x4x2_t _r50_02461357 = vld2q_f32(r5b);
                    float32x4x2_t _r50nx2 = vld2q_f32(r5b + 8);
                    float32x4_t _r5_8101214 = _r50nx2.val[0];
                    float32x4_t _r5_9111315 = _r50nx2.val[1];
                    float32x4_t _r50 = _r50_02461357.val[0];
                    float32x4_t _r51 = _r50_02461357.val[1];
                    float32x4_t _r52 = vextq_f32(_r50, _r5_8101214, 1);
                    float32x4_t _r53 = vextq_f32(_r51, _r5_9111315, 1);
                    float32x4_t _r54 = vextq_f32(_r50, _r5_8101214, 2);
                    float32x4_t _r55 = vextq_f32(_r51, _r5_9111315, 2);
                    float32x4_t _r56 = vextq_f32(_r50, _r5_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r50, _k35363738, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r51, _k35363738, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r52, _k35363738, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r53, _k35363738, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r54, _k39404142, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r55, _k39404142, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r56, _k39404142, 2);

                    float32x4_t _k42434445 = vld1q_f32(k6);
                    float32x4_t _k46474849 = vld1q_f32(k6 + 4);

                    float32x4x2_t _r60_02461357 = vld2q_f32(r6b);
                    float32x4x2_t _r60nx2 = vld2q_f32(r6b + 8);
                    float32x4_t _r6_8101214 = _r60nx2.val[0];
                    float32x4_t _r6_9111315 = _r60nx2.val[1];
                    float32x4_t _r60 = _r60_02461357.val[0];
                    float32x4_t _r61 = _r60_02461357.val[1];
                    float32x4_t _r62 = vextq_f32(_r60, _r6_8101214, 1);
                    float32x4_t _r63 = vextq_f32(_r61, _r6_9111315, 1);
                    float32x4_t _r64 = vextq_f32(_r60, _r6_8101214, 2);
                    float32x4_t _r65 = vextq_f32(_r61, _r6_9111315, 2);
                    float32x4_t _r66 = vextq_f32(_r60, _r6_8101214, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r60, _k42434445, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r61, _k42434445, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r62, _k42434445, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r63, _k42434445, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r64, _k46474849, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r65, _k46474849, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r66, _k46474849, 2);

                    vst1q_f32(outptrb, _sum);

                    r0b += 8;
                    r1b += 8;
                    r2b += 8;
                    r3b += 8;
                    r4b += 8;
                    r5b += 8;
                    r6b += 8;
                    outptrb += 4;
                }

                r0b += tailstep;
                r1b += tailstep;
                r2b += tailstep;
                r3b += tailstep;
                r4b += tailstep;
                r5b += tailstep;
                r6b += tailstep;
            }






// Line 781
// Assembly

// ************************** more than 30 operands used ***********************************
// ************************** place under #if __clang__  ***********************************

            i = 0;
            for (; i < outh; i++)
            {
                nn = outw >> 2;

                if (nn > 0)
                {
                asm volatile(
                    // v0:  input / final output
                    // v1 v2: = _ri0/_ri1  first 
                    // v3 v4: =                  then _r0_8101214/_r0_9111315
                    // v5 = ri2 / ri4 / ri6
                    // v6 = ri3 / ri5
                    // v9 = intermediate sum register
                    "0:                                        \n"                    
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1]                  \n"

                    //i = 1
                    "prfm       pldl1keep, [%2, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%2]           \n" // v1  v2 = _r00  _r01
                    "add        %2, %2, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%2]           \n" // v3  v4 = _r0_8101214 / _r0_9111315     
                    "fmul       v9.4s, v1.4s, %18.s[0]         \n" // *+ _r00                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n" // v5 = _r02
                    "fmla       v0.4s, v2.4s, %18.s[1]         \n" // *+ _r01
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n" // v6 = _r03
                    "fmla       v9.4s, v5.4s, %18.s[2]         \n" // *+ _r02
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n" // v5 = _r04
                    "fmla       v0.4s, v6.4s, %18.s[3]         \n" // *+ _r03
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n" // v6 = _r05
                    "fmla       v9.4s, v5.4s, %19.s[0]         \n" // *+ _r04                    
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n" // v5 = _r06
                    "fmla       v0.4s, v6.4s, %19.s[1]         \n" // *+ _r05
                    "fmla       v9.4s, v5.4s, %19.s[2]         \n" // *+ _r06

                    //i = 2
                    "prfm       pldl1keep, [%3, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%3]           \n"
                    "add        %3, %3, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%3]           \n"    
                    "fmla       v9.4s, v1.4s, %20.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %20.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %20.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %20.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %21.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %21.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %21.s[2]         \n"

                    //i = 3
                    "prfm       pldl1keep, [%4, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%4]           \n"
                    "add        %4, %4, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%4]           \n"    
                    "fmla       v9.4s, v1.4s, %22.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %22.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %22.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %22.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %23.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %23.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %23.s[2]         \n"

                    //i = 4
                    "prfm       pldl1keep, [%5, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%5]           \n"
                    "add        %5, %5, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%5]           \n"    
                    "fmla       v9.4s, v1.4s, %24.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %24.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %24.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %24.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %25.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %25.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %25.s[2]         \n"

                    //i = 5
                    "prfm       pldl1keep, [%6, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%6]           \n"
                    "add        %6, %6, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%6]           \n"    
                    "fmla       v9.4s, v1.4s, %26.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %26.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %26.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %26.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %27.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %27.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %27.s[2]         \n"

                    //i = 6
                    "prfm       pldl1keep, [%7, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%7]           \n"
                    "add        %7, %7, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%7]           \n"    
                    "fmla       v9.4s, v1.4s, %28.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %28.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %28.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %28.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %29.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %29.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %29.s[2]         \n"

                    //i = 7
                    "prfm       pldl1keep, [%8, #512]          \n"
                    "ld2        {v1.4s, v2.4s}, [%8]           \n"
                    "add        %8, %8, #32                    \n"
                    "ld2        {v3.4s, v4.4s}, [%8]           \n"    
                    "fmla       v9.4s, v1.4s, %30.s[0]         \n"                       
                    "ext        v5.16b, v1.16b, v3.16b, #4     \n"
                    "fmla       v0.4s, v2.4s, %30.s[1]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #4     \n"
                    "fmla       v9.4s, v5.4s, %30.s[2]         \n"
                    "ext        v5.16b, v1.16b, v3.16b, #8     \n"
                    "fmla       v0.4s, v6.4s, %30.s[3]         \n"
                    "ext        v6.16b, v2.16b, v4.16b, #8     \n"
                    "fmla       v9.4s, v5.4s, %31.s[0]         \n"                   
                    "ext        v5.16b, v1.16b, v3.16b, #12    \n"
                    "fmla       v0.4s, v6.4s, %31.s[1]         \n"
                    "fmla       v9.4s, v5.4s, %31.s[2]         \n"

                    "fadd       v0.4s, v0.4s, v9.4s            \n"                    
                    "st1        {v0.4s}, [%1], #16             \n"                    
                    "subs       %w0, %w0, #1                   \n"
                    "bne        0b                             \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4),         // %6
                      "=r"(r5),         // %7
                      "=r"(r6)          // %8
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(r5),
                      "8"(r6),
                      "w"(_k0123),     // %18
                      "w"(_k4567),     // %19
                      "w"(_k78910),    // %20
                      "w"(_k11121314), // %21
                      "w"(_k14151617), // %22
                      "w"(_k18192021), // %23
                      "w"(_k21222324), // %24
                      "w"(_k25262728), // %25
                      "w"(_k28293031), // %26
                      "w"(_k32333435), // %27
                      "w"(_k35363738), // %28
                      "w"(_k39404142), // %29
                      "w"(_k42434445), // %30
                      "w"(_k46474849)  // %31
                    : "cc", "memory","v0", "v1", "v2", "v3", "v4", "v5", "v6", "v9"
                );
                }    


                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
                r5 += tailstep;
                r6 += tailstep;
            }




  EXPECT_NE(outptr, outptrb);
  for (int i=0; i<outw*outh; i++)
  {
      EXPECT_NEAR(*(out3+i), *(out4+i),1e-5);
//    printf("the %d of cc is : %f\n", i, *(out3+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out4+i));
  }


  delete []out;
  delete []out2;
  delete []out3;
  delete []out4;

}

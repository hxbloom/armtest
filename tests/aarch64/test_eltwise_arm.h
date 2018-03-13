#pragma once
#include <arm_neon.h>
#include "gtest/gtest.h"
#include "math.h"


TEST(aarch64_assembly, eltwise) 
{
  int nn = 5;
  int nn2 = 5;
  float mean = 5.f;
  float norm = 10.f;
  float coeff0 = 10;
  float coeff1 = 5;
  float coeff = 2;  
  
  float c0[20] __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float c1[20] __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float c2[20] __attribute__((aligned(16))) = {3,3,3,3,3,3,3,3,3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
  
  float d0[20] __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float d1[20] __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float d2[20] __attribute__((aligned(16))) = {3,3,3,3,3,3,3,3,3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
  
  float e0[20] __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8, 9,10,11,12,13,14,15,16,17,18,19,20};
  float e1[20] __attribute__((aligned(16))) = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};
  float e2[20] __attribute__((aligned(16))) = {3,3,3,3,3,3,3,3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
  
  // For Product
  float *cc = c0;
  float *cc1 = c1;
  float *cc2 = c2;
  float *cc_out = new float[20]();
  float *dd = d0;
  float *dd1 = d1;
  float *dd2 = d2;
  float *dd_out = new float[20]();
  
  // Sum, no coefficient
  float *ee = c0;
  float *ee1 = c1;
  float *ee2 = c2;
  float *ee_out = new float[20]();
  float *ff = d0;
  float *ff1 = d1;
  float *ff2 = d2;
  float *ff_out = new float[20]();
  
  // Sum, with coefficient
  float *gg = c0;
  float *gg1 = c1;
  float *gg2 = c2;
  float *gg_out = new float[20]();
  float *hh = d0;
  float *hh1 = d1;
  float *hh2 = d2;
  float *hh_out = new float[20]();
  
  // Max
  float *ii = e0;
  float *ii1 = e1;
  float *ii2 = e2;
  float *ii_out = new float[20]();
  float *jj = e0;
  float *jj1 = e1;
  float *jj2 = e2;
  float *jj_out = new float[20]();
  


// ****** Product
// Line 58
// first blob
  float *ptr = (float*)cc;
  float *ptr1 = (float*)cc1;
  float *outptr = (float*)cc_out;
  float *ptr2 = (float*)dd;
  float *ptr21 = (float*)dd1;
  float *outptr2 = (float*)dd_out;
  
            for (; nn2>0; nn2--)
            {
                float32x4_t _ptr2 = vld1q_f32(ptr2);
                float32x4_t _ptr21 = vld1q_f32(ptr21);
                float32x4_t _p = vmulq_f32(_ptr2, _ptr21);
                vst1q_f32(outptr2, _p);

                ptr2 += 4;
                ptr21 += 4;
                outptr2 += 4;
            }

// first blob: assembly
            asm volatile(
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "prfm       pldl1keep, [%2, #128] \n"
                "ld1        {v0.4s}, [%1], #16    \n"
                "ld1        {v1.4s}, [%2], #16    \n"
                "fmul       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%3], #16    \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "v0", "v1"
            );

// Product
// Line 123
// second+ blobs
  nn = 5;
  nn2 = 5;
  ptr = (float*)cc2;
  outptr = (float*)cc_out;
  ptr2 = (float*)dd2;
  outptr2 = (float*)dd_out;
  
                for (; nn2>0; nn2--)
                {
                    float32x4_t _ptr = vld1q_f32(ptr2);
                    float32x4_t _p = vld1q_f32(outptr2);
                    _p = vmulq_f32(_ptr, _p);
                    vst1q_f32(outptr2, _p);

                    ptr2 += 4;
                    outptr2 += 4;
                }
  
// second+ blobs: assembly
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2]         \n"
                    "fmul       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%2], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );





// ****** SUM without coefficients
// Line 189
// first blob
  nn = 5;
  nn2 = 5;
  ptr = (float*)ee;
  ptr1 = (float*)ee1;
  outptr = (float*)ee_out;
  ptr2 = (float*)ff;
  ptr21 = (float*)ff;
  outptr2 = (float*)ff_out;
  
            for (; nn2>0; nn2--)
            {
                float32x4_t _ptr2 = vld1q_f32(ptr2);
                float32x4_t _ptr21 = vld1q_f32(ptr21);
                float32x4_t _p = vaddq_f32(_ptr2, _ptr21);
                vst1q_f32(outptr2, _p);

                ptr2 += 4;
                ptr21 += 4;
                outptr2 += 4;
            }

// first blob: assembly
            asm volatile(
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "prfm       pldl1keep, [%2, #128] \n"
                "ld1        {v0.4s}, [%1], #16    \n"
                "ld1        {v1.4s}, [%2], #16    \n"
                "fadd       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%3], #16    \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "v0", "v1"
            );


// SUM without coefficients
// Line 254
// second+ blobs
  nn = 5;
  nn2 = 5;
  ptr = (float*)ee2;
  outptr = (float*)ee_out;
  ptr2 = (float*)ff2;
  outptr2 = (float*)ff_out;
  
                for (; nn2>0; nn2--)
                {
                    float32x4_t _ptr = vld1q_f32(ptr2);
                    float32x4_t _p = vld1q_f32(outptr2);
                    _p = vaddq_f32(_ptr, _p);
                    vst1q_f32(outptr2, _p);

                    ptr2 += 4;
                    outptr2 += 4;
                }
  
// second+ blobs: assembly
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2]         \n"
                    "fadd       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%2], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );





// ****** SUM with coefficients
                float32x4_t _coeff0 = vdupq_n_f32(coeff0);
                float32x4_t _coeff1 = vdupq_n_f32(coeff1);
// Line 324
// first blob
  nn = 5;
  nn2 = 5;
  ptr = (float*)gg;
  ptr1 = (float*)gg1;
  outptr = (float*)gg_out;
  ptr2 = (float*)hh;
  ptr21 = (float*)hh;
  outptr2 = (float*)hh_out;

                for (; nn2>0; nn2--)
                {
                    float32x4_t _ptr = vld1q_f32(ptr2);
                    float32x4_t _ptr1 = vld1q_f32(ptr21);
                    float32x4_t _p = vmulq_f32(_ptr, _coeff0);
                    _p = vmlaq_f32(_p, _ptr1, _coeff1);
                    vst1q_f32(outptr2, _p);

                    ptr2 += 4;
                    ptr21 += 4;
                    outptr2 += 4;
                }

// first blob: assembly
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2], #16    \n"
                    "fmul       v0.4s, v0.4s, %8.4s   \n"
                    "fmla       v0.4s, v1.4s, %9.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%3], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr),
                      "w"(_coeff0), // %8
                      "w"(_coeff1)  // %9
                    : "cc", "memory", "v0", "v1"
                );

// SUM with coefficients
                    float32x4_t _coeff = vdupq_n_f32(coeff);
// Line 395
// second+ blob
  nn = 5;
  nn2 = 5;
  ptr = (float*)gg2;
  outptr = (float*)gg_out;
  ptr2 = (float*)hh2;
  outptr2 = (float*)hh_out;

                    for (; nn2>0; nn2--)
                    {
                        float32x4_t _ptr = vld1q_f32(ptr2);
                        float32x4_t _p = vld1q_f32(outptr2);
                        _p = vmlaq_f32(_p, _ptr, _coeff);
                        vst1q_f32(outptr2, _p);

                        ptr2 += 4;
                        outptr2 += 4;
                    }

// second+ blob: assembly
                    asm volatile(
                        "0:                               \n"
                        "prfm       pldl1keep, [%1, #128] \n"
                        "prfm       pldl1keep, [%2, #128] \n"
                        "ld1        {v0.4s}, [%1], #16    \n"
                        "ld1        {v1.4s}, [%2]         \n"
                        "fmla       v1.4s, v0.4s, %6.4s   \n"
                        "subs       %w0, %w0, #1          \n"
                        "st1        {v1.4s}, [%2], #16    \n"
                        "bne        0b                    \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr),
                          "w"(_coeff)   // %6
                        : "cc", "memory", "v0", "v1"
                    );

  
  


// ****** MAX
// Line 461
// first blob
  nn = 5;
  nn2 = 5;
  ptr = (float*)ii;
  ptr1 = (float*)ii1;
  outptr = (float*)ii_out;
  ptr2 = (float*)jj;
  ptr21 = (float*)jj1;
  outptr2 = (float*)jj_out;
  
            for (; nn2>0; nn2--)
            {
                float32x4_t _ptr = vld1q_f32(ptr2);
                float32x4_t _ptr1 = vld1q_f32(ptr21);
                float32x4_t _p = vmaxq_f32(_ptr, _ptr1);
                vst1q_f32(outptr2, _p);

                ptr2 += 4;
                ptr21 += 4;
                outptr2 += 4;
            }

// first blob: assembly
            asm volatile(
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "prfm       pldl1keep, [%2, #128] \n"
                "ld1        {v0.4s}, [%1], #16    \n"
                "ld1        {v1.4s}, [%2], #16    \n"
                "fmax       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%3], #16    \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "v0", "v1"
            );

// MAX
// Line 526
// second+ blob
  nn = 5;
  nn2 = 5;
  ptr = (float*)ii2;
  outptr = (float*)ii_out;
  ptr2 = (float*)jj2;
  outptr2 = (float*)jj_out;

                for (; nn2>0; nn2--)
                {
                    float32x4_t _ptr = vld1q_f32(ptr2);
                    float32x4_t _p = vld1q_f32(outptr2);
                    _p = vmaxq_f32(_ptr, _p);
                    vst1q_f32(outptr2, _p);

                    ptr2 += 4;
                    outptr2 += 4;
                }

// second+ blob: assembly
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2]         \n"
                    "fmax       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%2], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );

  
  
  for (int i=0; i<20; i++)
  { 
    EXPECT_EQ(*(cc_out+i), *(dd_out+i));
    EXPECT_EQ(*(cc_out+i), (i+1.f)*(i+1.f)*3);
    
    EXPECT_EQ(*(ee_out+i), *(ff_out+i));
    EXPECT_EQ(*(ee_out+i), (i+1.f)+(i+1.f)+3);
    
    EXPECT_EQ(*(gg_out+i), *(hh_out+i));
    EXPECT_EQ(*(gg_out+i), coeff0*(i+1.f)+coeff1*(i+1.f)+coeff*3.f);
    
    EXPECT_EQ(*(ii_out+i), *(jj_out+i));
    EXPECT_EQ(*(ii_out+i), fmaxf(i+2.f,3.f));
  }

  delete []cc_out;
  delete []dd_out;
  delete []ee_out;
  delete []ff_out;
  delete []gg_out;
  delete []hh_out;
  delete []ii_out;
  delete []jj_out;
}

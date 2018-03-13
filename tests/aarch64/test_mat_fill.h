#pragma once
#include <arm_neon.h>
#include "gtest/gtest.h"

/*
 * aarch64 version of mat_fill operation in src/mat.h
 * 
 */

TEST(aarch64_assembly, mat_fill) 
{
  int nn = 5;
  int nn2 = 5;
  float a = 99.f;
  
  float c[20] __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float d[20] __attribute__((aligned(16))) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float *cc = c;
  float *dd = d;

  float32x4_t _c = vdupq_n_f32(a);

  float *ptr = (float*)cc;
  float *ptr2 = (float*)dd;

  for (; nn2>0; nn2--)
  {
    vst1q_f32(ptr2, _c);
    ptr2 += 4;
  }
  
  asm volatile (
        "0:                             \n"
        "subs       %w0, %w0, #1        \n"
        "st1        {%4.4s}, [%1], #16  \n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(ptr)     // %1
        : "0"(nn),
          "1"(ptr),
          "w"(_c)       // %4
        : "cc", "memory"
  );
  
  for (int i=0; i<20; i++)
  {
    EXPECT_EQ(*(cc+i), a);
    EXPECT_EQ(*(cc+i), a);
  }

}

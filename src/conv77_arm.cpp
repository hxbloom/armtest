#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>


#define ENABLE_BIAS 1

#ifdef ENABLE_BIAS
#include <algorithm>
#endif

int main()
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
  float ker1[49] __attribute__((aligned(16)))  = {20,1,1,1,1,1,1,    1,1,1,1,1,1,1, \
                                                  1,1,1,1,1,1,1,    1,1,1,1,1,1,1, \
                                                  1,1,1,1,1,1,1,    1,1,1,1,1,1,1, \
                                                  1,1,1,1,1,1,1                    }; // Test kernel for main part
  float* kernel0 = ker1;

  const float* k0 = kernel0;
  const float* k1 = kernel0 + 7;
  const float* k2 = kernel0 + 14; 
  const float* k3 = kernel0 + 21;
  const float* k4 = kernel0 + 28;
  const float* k5 = kernel0 + 35;
  const float* k6 = kernel0 + 42;
  
  
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
  const float bias0 = 10000.f;
    
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





// Line 6863
// Case: when stride = 2, compute 1*h at a time
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

// Line 6908
// Case: when stride = 2, compute 1*h at a time
// Assembly

            i = 0;
            for (; i < outh; i++)
            {
                nn = outw >> 2;

                if (nn > 0)
                {
                asm volatile(
                    // v0 as final output
                    // v12 = rx2
                    // v13 v14 = intermediate sum register

                    "0:                                        \n"                    
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1]                  \n"
                    
                    "fmul       v1.4s, v0.4s, %18.s[0]         \n"
                    "st1        {v1.4s}, [%1], #16             \n"                    
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
                    : "cc", "memory","v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16"  // remaining Neon Register = 31-14 = 17
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


  for (int i=0; i<outw*outh; i++)
  {
    printf("the %d of cc is : %f\n", i, *(out+i));
    printf("-----------------the %d of dd is : %f\n", i, *(out2+i));
  }


  delete []out;
  delete []out2;

  return 0;
}

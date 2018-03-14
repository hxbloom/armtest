#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>

#define ENABLE_BIAS 1

#ifdef ENABLE_BIAS
#include <algorithm>
#endif

int main()
{
  // Test the assembly part only
  // Input is set to 1  0\11 respectively, so that outw=outh=9, let outh to be odd number.
  // Test is for 1 group only. (group is not considered)
  int w = 12;
  int h = 13;
  int outw = w-4;
  int outh = h-4;
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
  float ker1[25] __attribute__((aligned(16)))  = {1,1,1,1,1,1,1,1,1, \
                                                  1,1,1,1,1,1,1,1,1, \
                                                  1,1,1,1,1,1,1     }; // Test kernel for stride 1
  float* kernel0 = ker1;
  
  const float bias0 = 10000.f;
  
  float *r0 = c0;
  float *r1 = c0 + w;
  float *r2 = c0 + w * 2;
  float *r3 = c0 + w * 3;
  float* r4 = c0 + w * 4;
  float* r5 = c0 + w * 5;
  
  float *r02 = c0;
  float *r12 = c0 + w;
  float *r22 = c0 + w * 2;
  float *r32 = c0 + w * 3;
  float* r42 = c0 + w * 4;
  float* r52 = c0 + w * 5;
  float *out  = new float[outw*outh]();
  float *out2 = new float[outw*outh]();
#ifdef ENABLE_BIAS
  // Fill the bias
  
  std::fill_n(out, outw*outh, bias0); 
  std::fill_n(out2, outw*outh, bias0); 
#endif

  float* outptr = out;
  float* outptr2 = outptr + outw;
  float* outptrb = out2;
  float* outptr2b = outptrb + outw;

        // Shared part of both intrinsics and assembly
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k4567 = vld1q_f32(kernel0+4);
            float32x4_t _k891011 = vld1q_f32(kernel0+8);
            float32x4_t _k12131415 = vld1q_f32(kernel0+12);
            float32x4_t _k16171819 = vld1q_f32(kernel0+16);
            float32x4_t _k20212223 = vld1q_f32(kernel0+20);
            float32x4_t _k24242424 = vdupq_n_f32(kernel0[24]);

        float32x4_t _bias0 = vdupq_n_f32(bias0);


// Line 86
// Case: when stride = 1, compute 2*h at a time
// Intrinsic

            int i = 0;
#if __aarch64__
            for (; i+1 < outh; i+=2)
            {
                nn2 = outw >> 2;

                for (; nn2>0; nn2--)
                {
                    float32x4_t _sum = vld1q_f32(outptrb);
                    float32x4_t _sum2 = vld1q_f32(outptr2b);

                    float32x4_t _r00 = vld1q_f32(r02);
                    float32x4_t _r04 = vld1q_f32(r02 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r04, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r04, 2);
                    float32x4_t _r03 = vextq_f32(_r00, _r04, 3);

                    float32x4_t _r10 = vld1q_f32(r12);
                    float32x4_t _r14 = vld1q_f32(r12 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r14, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r14, 2);
                    float32x4_t _r13 = vextq_f32(_r10, _r14, 3);

                    float32x4_t _r20 = vld1q_f32(r22);
                    float32x4_t _r24 = vld1q_f32(r22 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r24, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r24, 2);
                    float32x4_t _r23 = vextq_f32(_r20, _r24, 3);

                    float32x4_t _r30 = vld1q_f32(r32);
                    float32x4_t _r34 = vld1q_f32(r32 + 4);
                    float32x4_t _r31 = vextq_f32(_r30, _r34, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r34, 2);
                    float32x4_t _r33 = vextq_f32(_r30, _r34, 3);

                    float32x4_t _r40 = vld1q_f32(r42);
                    float32x4_t _r44 = vld1q_f32(r42 + 4);
                    float32x4_t _r41 = vextq_f32(_r40, _r44, 1);
                    float32x4_t _r42 = vextq_f32(_r40, _r44, 2);
                    float32x4_t _r43 = vextq_f32(_r40, _r44, 3);

                    float32x4_t _r50 = vld1q_f32(r52);
                    float32x4_t _r54 = vld1q_f32(r52 + 4);
                    float32x4_t _r51 = vextq_f32(_r50, _r54, 1);
                    float32x4_t _r52 = vextq_f32(_r50, _r54, 2);
                    float32x4_t _r53 = vextq_f32(_r50, _r54, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k4567, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k4567, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k891011, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k891011, 1);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k891011, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k891011, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k12131415, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k12131415, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k12131415, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k12131415, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k16171819, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k16171819, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k16171819, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k16171819, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k20212223, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k20212223, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k20212223, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k20212223, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k24242424, 0);

                    _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k0123, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r11, _k0123, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k0123, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r13, _k0123, 3);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r14, _k4567, 0);

                    _sum2 = vfmaq_laneq_f32(_sum2, _r20, _k4567, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k4567, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r22, _k4567, 3);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r23, _k891011, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r24, _k891011, 1);

                    _sum2 = vfmaq_laneq_f32(_sum2, _r30, _k891011, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r31, _k891011, 3);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r32, _k12131415, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r33, _k12131415, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r34, _k12131415, 2);

                    _sum2 = vfmaq_laneq_f32(_sum2, _r40, _k12131415, 3);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r41, _k16171819, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r42, _k16171819, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r43, _k16171819, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r44, _k16171819, 3);

                    _sum2 = vfmaq_laneq_f32(_sum2, _r50, _k20212223, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r51, _k20212223, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r52, _k20212223, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r53, _k20212223, 3);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r54, _k24242424, 0);

                    vst1q_f32(outptrb, _sum);
                    vst1q_f32(outptr2b, _sum2);

                    r02 += 4;
                    r12 += 4;
                    r22 += 4;
                    r32 += 4;
                    r42 += 4;
                    r52 += 4;
                    outptrb += 4;
                    outptr2b += 4;
                }
                
                r02 += 4 + w;
                r12 += 4 + w;
                r22 += 4 + w;
                r32 += 4 + w;
                r42 += 4 + w;
                r52 += 4 + w;

                outptrb += outw;
                outptr2b += outw;
            }
#endif // __aarch64__


// Case: when stride = 1, compute the last h (if exists) at a time
// Intrinsic
// Line 558
#if __aarch64__
            for (; i < outh; i++)
            {

                nn2 = outw >> 2;

                for (; nn>0; nn--)
                {
                    float32x4_t _sum = vld1q_f32(outptrb);

                    float32x4_t _r00 = vld1q_f32(r02);
                    float32x4_t _r04 = vld1q_f32(r02 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r04, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r04, 2);
                    float32x4_t _r03 = vextq_f32(_r00, _r04, 3);

                    float32x4_t _r10 = vld1q_f32(r12);
                    float32x4_t _r14 = vld1q_f32(r12 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r14, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r14, 2);
                    float32x4_t _r13 = vextq_f32(_r10, _r14, 3);

                    float32x4_t _r20 = vld1q_f32(r22);
                    float32x4_t _r24 = vld1q_f32(r22 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r24, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r24, 2);
                    float32x4_t _r23 = vextq_f32(_r20, _r24, 3);

                    float32x4_t _r30 = vld1q_f32(r32);
                    float32x4_t _r34 = vld1q_f32(r32 + 4);
                    float32x4_t _r31 = vextq_f32(_r30, _r34, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r34, 2);
                    float32x4_t _r33 = vextq_f32(_r30, _r34, 3);

                    float32x4_t _r40 = vld1q_f32(r42);
                    float32x4_t _r44 = vld1q_f32(r42 + 4);
                    float32x4_t _r41 = vextq_f32(_r40, _r44, 1);
                    float32x4_t _r42 = vextq_f32(_r40, _r44, 2);
                    float32x4_t _r43 = vextq_f32(_r40, _r44, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k4567, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k4567, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k891011, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k891011, 1);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k891011, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k891011, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k12131415, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k12131415, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k12131415, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k12131415, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k16171819, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k16171819, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k16171819, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k16171819, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k20212223, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k20212223, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k20212223, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k20212223, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k24242424, 0);

                    vst1q_f32(outptrb, _sum);

                    r02 += 4;
                    r12 += 4;
                    r22 += 4;
                    r32 += 4;
                    r42 += 4;
                    outptrb += 4;
                }

                r02 += 4;
                r12 += 4;
                r22 += 4;
                r32 += 4;
                r42 += 4;

            }
#endif // __aarch64__

// Case: when stride = 1, compute 2*h at a time
// Assembly
// Line 202
#if __aarch64__
            i = 0;   
            for (; i+1 < outh; i+=2)
            {
                nn = outw >> 2;

                if (nn > 0)
                {
                asm volatile(
                    // v11 = rx1 / rx3
                    // v12 = rx2
                    // v13 v14 = intermediate sum register
                    
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v7.4s}, [%1]                  \n"// v7 = out

                    "0:                                        \n"

                    "prfm       pldl1keep, [%2, #128]          \n"
                    "ld1        {v8.4s}, [%2]                  \n"// v8 = out2

                    // r1
                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld1        {v9.4s, v10.4s}, [%4]          \n"// v9 v10 = r10 r14
                    "add        %4, %4, #16                    \n"

                    "ext        v11.16b, v9.16b, v10.16b, #4   \n" //r11
                    "fmul       v13.4s, v9.4s, %19.s[1]        \n"
                    "fmla       v8.4s,  v9.4s, %18.s[0]        \n"

                    "ext        v12.16b, v9.16b, v10.16b, #8   \n" //r12
                    "fmla       v7.4s,  v11.4s, %19.s[2]       \n"
                    "fmul       v14.4s, v11.4s, %18.s[1]       \n"

                    "ext        v11.16b, v9.16b, v10.16b, #12  \n" //r13
                    "fmla       v13.4s, v12.4s, %19.s[3]       \n"
                    "fmla       v8.4s,  v12.4s, %18.s[2]       \n"

                    "fmla       v7.4s,  v11.4s, %20.s[0]       \n"
                    "fmla       v14.4s, v11.4s, %18.s[3]       \n"

                    "prfm       pldl1keep, [%5, #256]          \n"

                    "fmla       v13.4s, v10.4s, %20.s[1]       \n"
                    "fmla       v8.4s,  v10.4s, %19.s[0]       \n"

                    // r2
                    "ld1        {v9.4s, v10.4s}, [%5]          \n"// v9 v10 = r20 r24
                    "add        %5, %5, #16                    \n"

                    "ext        v11.16b, v9.16b, v10.16b, #4   \n" //r21
                    "fmla       v7.4s,  v9.4s, %20.s[2]        \n"
                    "fmla       v14.4s, v9.4s, %19.s[1]        \n"                    

                    "ext        v12.16b, v9.16b, v10.16b, #8   \n" //r22
                    "fmla       v13.4s, v11.4s, %20.s[3]       \n"
                    "fmla       v8.4s,  v11.4s, %19.s[2]       \n"   
                    
                    "ext        v11.16b, v9.16b, v10.16b, #12  \n" //r23
                    "fmla       v7.4s,  v12.4s, %21.s[0]       \n"
                    "fmla       v14.4s, v12.4s, %19.s[3]       \n"   

                    "fmla       v13.4s, v11.4s, %21.s[1]       \n"
                    "fmla       v8.4s,  v11.4s, %20.s[0]       \n" 

                    "prfm       pldl1keep, [%6, #256]          \n"

                    "fmla       v7.4s,  v10.4s, %21.s[2]       \n"
                    "fmla       v14.4s, v10.4s, %20.s[1]       \n" 
                    
                    // r3
                    "ld1        {v9.4s, v10.4s}, [%6]          \n"// v9 v10 = r30 r34
                    "add        %6, %6, #16                    \n"

                    "ext        v11.16b, v9.16b, v10.16b, #4   \n" //r31
                    "fmla       v13.4s, v9.4s, %21.s[3]        \n"
                    "fmla       v8.4s,  v9.4s, %20.s[2]        \n"   

                    "ext        v12.16b, v9.16b, v10.16b, #8   \n" //r32
                    "fmla       v7.4s,  v11.4s, %22.s[0]       \n"
                    "fmla       v14.4s, v11.4s, %20.s[3]       \n"   

                    "ext        v11.16b, v9.16b, v10.16b, #12  \n" //r33
                    "fmla       v13.4s, v12.4s, %22.s[1]       \n"
                    "fmla       v8.4s,  v12.4s, %21.s[0]       \n"   

                    "fmla       v7.4s,  v11.4s, %22.s[2]       \n"
                    "fmla       v14.4s, v11.4s, %21.s[1]       \n"   

                    "prfm       pldl1keep, [%7, #256]          \n"

                    "fmla       v13.4s, v10.4s, %22.s[3]       \n"
                    "fmla       v8.4s,  v10.4s, %21.s[2]       \n"  

                    // r4
                    "ld1        {v9.4s, v10.4s}, [%7]          \n"// v9 v10 = r40 r44
                    "add        %7, %7, #16                    \n"

                    "ext        v11.16b, v9.16b, v10.16b, #4   \n" //r41
                    "fmla       v7.4s,  v9.4s, %23.s[0]        \n"
                    "fmla       v14.4s, v9.4s, %21.s[3]        \n"  

                    "ext        v12.16b, v9.16b, v10.16b, #8   \n" //r41
                    "fmla       v13.4s, v11.4s, %23.s[1]       \n"
                    "fmla       v8.4s,  v11.4s, %22.s[0]       \n"  

                    "ext        v11.16b, v9.16b, v10.16b, #12  \n" //r41
                    "fmla       v7.4s,  v12.4s, %23.s[2]       \n"
                    "fmla       v14.4s, v12.4s, %22.s[1]       \n"

                    "fmla       v13.4s, v11.4s, %23.s[3]       \n"
                    "fmla       v8.4s,  v11.4s, %22.s[2]       \n"

                    "prfm       pldl1keep, [%3, #256]          \n"

                    "fmla       v7.4s,  v10.4s, %24.s[0]       \n"
                    "fmla       v14.4s, v10.4s, %22.s[3]       \n"

                    // r0 and r5
                    "ld1        {v9.4s, v10.4s}, [%3]          \n"// v9 v10 = r00 r04
                    "add        %3, %3, #16                    \n"

                    "ext        v11.16b, v9.16b, v10.16b, #4   \n" //r01
                    "fmla       v13.4s, v11.4s, %18.s[1]       \n"

                    "ext        v12.16b, v9.16b, v10.16b, #8   \n" //r02
                    "fmla       v7.4s, v12.4s, %18.s[2]        \n"

                    "ext        v11.16b, v9.16b, v10.16b, #12  \n" //r03

                    "prfm       pldl1keep, [%8, #256]          \n"

                    "fmla       v13.4s, v11.4s, %18.s[3]       \n"

                    // r5
                    "ld1        {v11.4s, v12.4s}, [%8]         \n"// v11 v12 = r50 r54
                    "add        %8, %8, #16                    \n"

                    "fmla       v8.4s,  v11.4s, %23.s[0]       \n"
                    "fmla       v14.4s, v12.4s, %24.s[0]       \n"

                    "fmla       v7.4s,  v9.4s,  %18.s[0]       \n"
                    "fmla       v13.4s, v10.4s, %19.s[0]       \n"

                    "ext        v9.16b,  v11.16b, v12.16b, #4  \n" //r51
                    "ext        v10.16b, v11.16b, v12.16b, #8  \n" //r52

                    "fmla       v14.4s, v9.4s, %23.s[1]        \n"

                    "ext        v9.16b, v11.16b, v12.16b, #12  \n" //r53
                    "fmla       v8.4s, v10.4s, %23.s[2]        \n"

                    "fmla       v14.4s, v9.4s, %23.s[3]        \n"

                    "fadd       v7.4s, v7.4s, v13.4s           \n"

                    "st1        {v7.4s}, [%1], #16             \n"

                    "fadd       v8.4s, v8.4s, v14.4s           \n"

                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v7.4s}, [%1]                  \n"// v7 = out
                    "st1        {v8.4s}, [%2], #16             \n"

                    "subs       %w0, %w0, #1                   \n"
                    "bne        0b                             \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(outptr2),    // %2
                      "=r"(r0),         // %3
                      "=r"(r1),         // %4
                      "=r"(r2),         // %5
                      "=r"(r3),         // %6
                      "=r"(r4),         // %7
                      "=r"(r5)          // %8
                    : "0"(nn),
                      "1"(outptr),
                      "2"(outptr2),
                      "3"(r0),
                      "4"(r1),
                      "5"(r2),
                      "6"(r3),
                      "7"(r4),
                      "8"(r5),
                      "w"(_k0123),      // %18
                      "w"(_k4567),      // %19
                      "w"(_k891011),    // %20
                      "w"(_k12131415),  // %21
                      "w"(_k16171819),  // %22
                      "w"(_k20212223),  // %23
                      "w"(_k24242424)   // %24
                    : "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }

                r0 += 4 + w;
                r1 += 4 + w;
                r2 += 4 + w;
                r3 += 4 + w;
                r4 += 4 + w;
                r5 += 4 + w;

                outptr += outw;
                outptr2 += outw;
            }
#endif // __aarch64__
// Case: when stride = 1, compute the last h (if exists) at a time
// Assembly
// Line 634
            for (; i < outh; i++)
            {

                nn = outw >> 2;
#if __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "prfm       pldl1keep, [%2, #256]          \n"

                    "ld1        {v8.4s, v9.4s}, [%2]           \n"// _r00 = vld1q_f32(r0+j);
                    "add        %2, %2, #16                    \n"

                    "0:                                        \n"

                    "ld1        {v7.4s}, [%1]                  \n"// _sum = vld1q_f32(outptr+j);

                    "ext        v10.16b, v8.16b, v9.16b, #4    \n" //_r01
                    "ext        v11.16b, v8.16b, v9.16b, #8    \n" //_r02
                    "ext        v12.16b, v8.16b, v9.16b, #12   \n" //_r03

                    "fmla       v7.4s,   v8.4s, %14.s[0]       \n"
                    "fmul       v13.4s, v10.4s, %14.s[1]       \n"

                    "prfm       pldl1keep, [%3, #256]          \n"

                    "fmul       v14.4s, v11.4s, %14.s[2]       \n"
                    "fmul       v15.4s, v12.4s, %14.s[3]       \n"
                    "fmla       v7.4s,   v9.4s, %15.s[0]       \n"

                    "ld1        {v8.4s, v9.4s}, [%3]           \n"
                    "add        %3, %3, #16                    \n"
                    "ext        v10.16b, v8.16b, v9.16b, #4    \n" //_r11
                    "ext        v11.16b, v8.16b, v9.16b, #8    \n" //_r12
                    "ext        v12.16b, v8.16b, v9.16b, #12   \n" //_r13

                    "fmla       v7.4s,   v8.4s, %15.s[1]       \n"
                    "fmla       v13.4s, v10.4s, %15.s[2]       \n"

                    "prfm       pldl1keep, [%4, #256]          \n"

                    "fmla       v14.4s, v11.4s, %15.s[3]       \n"
                    "fmla       v15.4s, v12.4s, %16.s[0]       \n"
                    "fmla       v7.4s,   v9.4s, %16.s[1]       \n"

                    "ld1        {v8.4s, v9.4s}, [%4]           \n"
                    "add        %4, %4, #16                    \n"
                    "ext        v10.16b, v8.16b, v9.16b, #4    \n" //_r21
                    "ext        v11.16b, v8.16b, v9.16b, #8    \n" //_r22
                    "ext        v12.16b, v8.16b, v9.16b, #12   \n" //_r23

                    "fmla       v7.4s,   v8.4s, %16.s[2]       \n"
                    "fmla       v13.4s, v10.4s, %16.s[3]       \n"

                    "prfm       pldl1keep, [%5, #256]          \n"

                    "fmla       v14.4s, v11.4s, %17.s[0]       \n"
                    "fmla       v15.4s, v12.4s, %17.s[1]       \n"
                    "fmla       v7.4s,   v9.4s, %17.s[2]       \n"

                    "ld1        {v8.4s, v9.4s}, [%5]           \n"
                    "add        %5, %5, #16                    \n"
                    "ext        v10.16b, v8.16b, v9.16b, #4    \n" //_r31
                    "ext        v11.16b, v8.16b, v9.16b, #8    \n" //_r32
                    "ext        v12.16b, v8.16b, v9.16b, #12   \n" //_r33

                    "fmla       v7.4s,   v8.4s, %17.s[3]       \n"
                    "fmla       v13.4s, v10.4s, %18.s[0]       \n"

                    "prfm       pldl1keep, [%6, #256]          \n"

                    "fmla       v14.4s, v11.4s, %18.s[1]       \n"
                    "fmla       v15.4s, v12.4s, %18.s[2]       \n"
                    "fmla       v7.4s,   v9.4s, %18.s[3]       \n"

                    "ld1        {v8.4s, v9.4s}, [%6]           \n"
                    "add        %6, %6, #16                    \n"
                    "ext        v10.16b, v8.16b, v9.16b, #4    \n" //_r41
                    "ext        v11.16b, v8.16b, v9.16b, #8    \n" //_r42
                    "ext        v12.16b, v8.16b, v9.16b, #12   \n" //_r43

                    "fmla       v7.4s,   v8.4s, %19.s[0]       \n"
                    "fmla       v13.4s, v10.4s, %19.s[1]       \n"
                    "fmla       v14.4s, v11.4s, %19.s[2]       \n"
                    "fmla       v15.4s, v12.4s, %19.s[3]       \n"
                    "fmla       v7.4s,   v9.4s, %20.s[0]       \n"

                    "fadd       v14.4s, v14.4s, v15.4s         \n"
                    "fadd       v7.4s,   v7.4s, v13.4s         \n"                    

                    "prfm       pldl1keep, [%2, #256]          \n"

                    "fadd       v7.4s,   v7.4s, v14.4s         \n"     

                    "ld1        {v8.4s, v9.4s}, [%2]           \n"
                    "add        %2, %2, #16                    \n"

                    "st1        {v7.4s}, [%1], #16             \n"

                    "prfm       pldl1keep, [%1, #128]          \n"

                    "subs       %w0, %w0, #1                   \n"
                    "bne        0b                             \n"

                    "sub        %2, %2, #16                    \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4)          // %6
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "w"(_k0123),      // %14
                      "w"(_k4567),      // %15
                      "w"(_k891011),    // %16
                      "w"(_k12131415),  // %17
                      "w"(_k16171819),  // %18
                      "w"(_k20212223),  // %19
                      "w"(_k24242424)   // %20
                    : "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }
#endif // __aarch64__

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;

            }
            
            
  outptr -= outw*outh;
  outptrb -= outw*outh;
  
  for (int i=0; i<outw*outh; i++)
  {
//    printf("the %d of cc is : %f\n", i, *(out+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out2+i));
  }
  
 

// Line 522
// Case: when stride = 2, compute 1*h at a time
// Input is set to 19\19 respectively, so that outw=outh=8, let outh to be odd number.
// Test is for 1 group only. (group is not considered)

  w = 19;
  h = 19;
  outw = (w-3)/2;
  outh = (w-3)/2;
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
  r4 = d0 + w * 4;
  r5 = d0 + w * 5;
  
  r02 = d0;
  r12 = d0 + w;
  r22 = d0 + w * 2;
  r32 = d0 + w * 3;
  r42 = d0 + w * 4;
  r52 = d0 + w * 5;
  float *out3  = new float[outw*outh]();
  float *out4  = new float[outw*outh]();

#ifdef ENABLE_BIAS
  // Fill the bias
  std::fill_n(out3, outw*outh, bias0); 
  std::fill_n(out4, outw*outh, bias0); 
#endif

  outptr = out3;
  outptrb = out4;

  float ker2[25] __attribute__((aligned(16)))  = {2,2,2,2,2,2,2,2,2, \
                                                  2,2,2,2,2,2,2,2,2, \
                                                  2,2,2,2,2,2,2      }; // Test kernel for stride 2
  kernel0 = ker2;
  
            _k0123 = vld1q_f32(kernel0);
            _k4567 = vld1q_f32(kernel0+4);
            _k891011 = vld1q_f32(kernel0+8);
            _k12131415 = vld1q_f32(kernel0+12);
            _k16171819 = vld1q_f32(kernel0+16);
            _k20212223 = vld1q_f32(kernel0+20);
            _k24242424 = vdupq_n_f32(kernel0[24]);

        _bias0 = vdupq_n_f32(bias0);

// Line 923
// Case: when stride = 2, compute 1*h at a time
// Intrinsic
#if __aarch64__
            i = 0;
            for (int i = 0; i < outh; i++)
            {
                nn2 = outw >> 2;

                for (; nn2>0; nn2--)
                {
                    float32x4_t _sum = vld1q_f32(outptrb);

                    float32x4x2_t _r00_02461357 = vld2q_f32(r02);
                    float32x4x2_t _r00nx2 = vld2q_f32(r02 + 8);
                    float32x4_t _r0_8101214 = _r00nx2.val[0];// 8 10 12 14
                    float32x4_t _r0_9111315 = _r00nx2.val[1];// 9 11 13 15
                    float32x4_t _r00 = _r00_02461357.val[0];// 0 2 4 6
                    float32x4_t _r01 = _r00_02461357.val[1];// 1 3 5 7
                    float32x4_t _r02 = vextq_f32(_r00, _r0_8101214, 1);// 2 4 6 8
                    float32x4_t _r03 = vextq_f32(_r01, _r0_9111315, 1);// 3 5 7 9
                    float32x4_t _r04 = vextq_f32(_r00, _r0_8101214, 2);// 4 6 8 10

                    float32x4x2_t _r10_02461357 = vld2q_f32(r12);
                    float32x4x2_t _r10nx2 = vld2q_f32(r12 + 8);
                    float32x4_t _r1_8101214 = _r10nx2.val[0];
                    float32x4_t _r1_9111315 = _r10nx2.val[1];
                    float32x4_t _r10 = _r10_02461357.val[0];
                    float32x4_t _r11 = _r10_02461357.val[1];
                    float32x4_t _r12 = vextq_f32(_r10, _r1_8101214, 1);
                    float32x4_t _r13 = vextq_f32(_r11, _r1_9111315, 1);
                    float32x4_t _r14 = vextq_f32(_r10, _r1_8101214, 2);

                    float32x4x2_t _r20_02461357 = vld2q_f32(r22);
                    float32x4x2_t _r20nx2 = vld2q_f32(r22 + 8);
                    float32x4_t _r2_8101214 = _r20nx2.val[0];
                    float32x4_t _r2_9111315 = _r20nx2.val[1];
                    float32x4_t _r20 = _r20_02461357.val[0];
                    float32x4_t _r21 = _r20_02461357.val[1];
                    float32x4_t _r22 = vextq_f32(_r20, _r2_8101214, 1);
                    float32x4_t _r23 = vextq_f32(_r21, _r2_9111315, 1);
                    float32x4_t _r24 = vextq_f32(_r20, _r2_8101214, 2);

                    float32x4x2_t _r30_02461357 = vld2q_f32(r32);
                    float32x4x2_t _r30nx2 = vld2q_f32(r32 + 8);
                    float32x4_t _r3_8101214 = _r30nx2.val[0];
                    float32x4_t _r3_9111315 = _r30nx2.val[1];
                    float32x4_t _r30 = _r30_02461357.val[0];
                    float32x4_t _r31 = _r30_02461357.val[1];
                    float32x4_t _r32 = vextq_f32(_r30, _r3_8101214, 1);
                    float32x4_t _r33 = vextq_f32(_r31, _r3_9111315, 1);
                    float32x4_t _r34 = vextq_f32(_r30, _r3_8101214, 2);

                    float32x4x2_t _r40_02461357 = vld2q_f32(r42);
                    float32x4x2_t _r40nx2 = vld2q_f32(r42 + 8);
                    float32x4_t _r4_8101214 = _r40nx2.val[0];
                    float32x4_t _r4_9111315 = _r40nx2.val[1];
                    float32x4_t _r40 = _r40_02461357.val[0];
                    float32x4_t _r41 = _r40_02461357.val[1];
                    float32x4_t _r42 = vextq_f32(_r40, _r4_8101214, 1);
                    float32x4_t _r43 = vextq_f32(_r41, _r4_9111315, 1);
                    float32x4_t _r44 = vextq_f32(_r40, _r4_8101214, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k4567, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k4567, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k891011, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k891011, 1);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k891011, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k891011, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k12131415, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k12131415, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k12131415, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k12131415, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k16171819, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k16171819, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k16171819, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k16171819, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k20212223, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k20212223, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k20212223, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k20212223, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k24242424, 0);

                    vst1q_f32(outptrb, _sum);

                    r02 += 8;
                    r12 += 8;
                    r22 += 8;
                    r32 += 8;
                    r42 += 8;
                    outptrb += 4;
                }

                r02 += tailstep;
                r12 += tailstep;
                r22 += tailstep;
                r32 += tailstep;
                r42 += tailstep;
            }
#endif // __aarch64__

// Line 1019
// Case: when stride = 2, compute 1*h at a time
// Assembly

            i = 0;
            for (int i = 0; i < outh; i++)
            {
                nn = outw >> 2;

                if (nn > 0)
                {
                asm volatile(
                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v8.4s, v9.4s}, [%2], #32      \n"// v8  = 0  2  4  6   q9  = 1  3  5  7

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld2        {v10.4s, v11.4s}, [%2]         \n"// v10 = 8 10 12 14   v11 = 9 11 13 15

                    "prfm       pldl1keep, [%1, #128]          \n"
                    "0:                                        \n"

                    "ld1        {v7.4s}, [%1]                  \n" // v7 = outptr

                    "ext        v12.16b, v8.16b, v10.16b, #4   \n" // v12 = 2 4 6 8
                    "ext        v11.16b, v9.16b, v11.16b, #4   \n" // v11 = 3 5 7 9
                    "ext        v10.16b, v8.16b, v10.16b, #8   \n" // v10 = 4 6 8 10

                    "fmla       v7.4s,  v8.4s, %14.s[0]        \n"
                    "fmul       v13.4s, v9.4s, %14.s[1]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"

                    "fmul       v14.4s, v12.4s, %14.s[2]       \n"
                    "fmul       v15.4s, v11.4s, %14.s[3]       \n"
                    "fmla       v7.4s,  v10.4s, %15.s[0]       \n"

                    "ld2        {v8.4s, v9.4s}, [%3], #32      \n"

                    "prfm       pldl1keep, [%3, #256]          \n"

                    "ld2        {v10.4s, v11.4s}, [%3]         \n"
                    "ext        v12.16b, v8.16b, v10.16b, #4   \n"
                    "ext        v11.16b, v9.16b, v11.16b, #4   \n"
                    "ext        v10.16b, v8.16b, v10.16b, #8   \n"

                    "fmla       v7.4s,  v8.4s, %15.s[1]        \n"
                    "fmla       v13.4s, v9.4s, %15.s[2]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"

                    "fmla       v14.4s, v12.4s, %15.s[3]       \n"
                    "fmla       v15.4s, v11.4s, %16.s[0]       \n"
                    "fmla       v7.4s,  v10.4s, %16.s[1]       \n"

                    "ld2        {v8.4s, v9.4s}, [%4], #32      \n"

                    "prfm       pldl1keep, [%4, #256]          \n"

                    "ld2        {v10.4s, v11.4s}, [%4]         \n"
                    "ext        v12.16b, v8.16b, v10.16b, #4   \n"
                    "ext        v11.16b, v9.16b, v11.16b, #4   \n"
                    "ext        v10.16b, v8.16b, v10.16b, #8   \n"

                    "fmla       v7.4s,  v8.4s, %16.s[2]        \n"
                    "fmla       v13.4s, v9.4s, %16.s[3]        \n"

                    "prfm       pldl1keep, [%5, #256]          \n"

                    "fmla       v14.4s, v12.4s, %17.s[0]       \n"
                    "fmla       v15.4s, v11.4s, %17.s[1]       \n"
                    "fmla       v7.4s,  v10.4s, %17.s[2]       \n"

                    "ld2        {v8.4s, v9.4s}, [%5], #32      \n"

                    "prfm       pldl1keep, [%5, #256]          \n"

                    "ld2        {v10.4s, v11.4s}, [%5]         \n"
                    "ext        v12.16b, v8.16b, v10.16b, #4   \n"
                    "ext        v11.16b, v9.16b, v11.16b, #4   \n"
                    "ext        v10.16b, v8.16b, v10.16b, #8   \n"

                    "fmla       v7.4s,  v8.4s, %17.s[3]        \n"
                    "fmla       v13.4s, v9.4s, %18.s[0]        \n"

                    "prfm       pldl1keep, [%6, #256]          \n"

                    "fmla       v14.4s, v12.4s, %18.s[1]       \n"
                    "fmla       v15.4s, v11.4s, %18.s[2]       \n"
                    "fmla       v7.4s,  v10.4s, %18.s[3]       \n"

                    "ld2        {v8.4s, v9.4s}, [%6], #32      \n"

                    "prfm       pldl1keep, [%6, #256]          \n"

                    "ld2        {v10.4s, v11.4s}, [%6]         \n"
                    "ext        v12.16b, v8.16b, v10.16b, #4   \n"
                    "ext        v11.16b, v9.16b, v11.16b, #4   \n"
                    "ext        v10.16b, v8.16b, v10.16b, #8   \n"

                    "fmla       v7.4s,   v8.4s, %19.s[0]       \n"
                    "fmla       v13.4s,  v9.4s, %19.s[1]       \n"
                    "fmla       v14.4s, v12.4s, %19.s[2]       \n"
                    "fmla       v15.4s, v11.4s, %19.s[3]       \n"
                    "fmla       v7.4s,  v10.4s, %20.s[0]       \n"                    

                    "prfm       pldl1keep, [%2, #256]          \n"

                    "ld2        {v8.4s, v9.4s}, [%2], #32      \n"

                    "fadd       v14.4s, v14.4s, v15.4s         \n"
                    "fadd       v7.4s,   v7.4s, v13.4s         \n"
                    
                    "prfm       pldl1keep, [%2, #256]          \n"

                    "fadd       v7.4s, v7.4s, v14.4s           \n"   

                    "ld2        {v10.4s, v11.4s}, [%2]         \n"
                    "st1        {v7.4s}, [%1], #16             \n"

                    "prfm       pldl1keep, [%1, #128]          \n"

                    "subs       %w0, %w0, #1                   \n"
                    "bne        0b                             \n"

                    "sub        %2, %2, #32                    \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4)          // %6
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "w"(_k0123),      // %14
                      "w"(_k4567),      // %15
                      "w"(_k891011),    // %16
                      "w"(_k12131415),  // %17
                      "w"(_k16171819),  // %18
                      "w"(_k20212223),  // %19
                      "w"(_k24242424)   // %20
                    : "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
            }



  outptr -= outw*outh;
  outptrb -= outw*outh;
  
  for (int i=0; i<outw*outh; i++)
  {
    printf("the %d of cc is : %f\n", i, *(out3+i));
    printf("-----------------the %d of dd is : %f\n", i, *(out4+i));
  }



  delete []out;
  delete []out2;
  delete []out3;
  delete []out4;
    
  return 0;
}

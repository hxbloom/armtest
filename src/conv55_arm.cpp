#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>

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
                                                  1,1,1,1,1,1,1     }; // Test kernel for main part
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


// Case: when stride = 1, compute 2*h at a time
// Assembly
// Line 202

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

// Case: when stride = 1, compute the last h (if exists) at a time
// Assembly
// Line 634
            for (; i < outh; i++)
            {

                nn = outw >> 2;
/*
                if (nn > 0)
                {
                asm volatile(
//                     "veor       q15, q15            \n"// _sum3 = 0;
//                    "pld        [%1, #128]          \n"
//                    "pld        [%2, #256]          \n"
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "prfm       pldl1keep, [%2, #256]          \n"

//                    "vld1.f32   {d16-d19}, [%2]     \n"// _r00 = vld1q_f32(r0+j);
//                    "add        %2, #16             \n"
                    "ld1        {v8.4s, v9.4s}, [%2]           \n"// _r00 = vld1q_f32(r0+j);
                    "add        %2, %2, #16                    \n"

                    "0:                                        \n"

//                    "vld1.f32   {d14-d15}, [%1]     \n"// _sum = vld1q_f32(outptr+j);
                    "ld1        {v7.4s}, [%1]                  \n"// _sum = vld1q_f32(outptr+j);
//                     "veor       q13, q13            \n"// _sum2 = 0;
//                     "veor       q14, q14            \n"// _sum3 = 0;

//                    "vext.32    q10, q8, q9, #1     \n"// _r01
//                    "vext.32    q11, q8, q9, #2     \n"// _r02
//                    "vext.32    q12, q8, q9, #3     \n"// _r03
                    "ext        v10.16b, v8.16b, v9.16b, #4    \n" //_r01
                    "ext        v11.16b, v8.16b, v9.16b, #8    \n" //_r02
                    "ext        v12.16b, v8.16b, v9.16b, #12   \n" //_r03

//                    "vmla.f32   q7, q8, %e14[0]     \n"
//                    "vmul.f32   q13, q10, %e14[1]   \n"
                    "fmla       v7.4s,   v8.4s, %14.s[0]       \n"
                    "fmul       v13.4s, v10.4s, %14.s[1]       \n"

                    "pld        [%3, #256]          \n"

                    "vmul.f32   q14, q11, %f14[0]   \n"
                    "vmul.f32   q15, q12, %f14[1]   \n"
                    "vmla.f32   q7, q9, %e15[0]     \n"

                    "vld1.f32   {d16-d19}, [%3]     \n"
                    "add        %3, #16             \n"
                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"
                    "vext.32    q12, q8, q9, #3     \n"

                    "vmla.f32   q7, q8, %e15[1]     \n"
                    "vmla.f32   q13, q10, %f15[0]   \n"

                    "pld        [%4, #256]          \n"

                    "vmla.f32   q14, q11, %f15[1]   \n"
                    "vmla.f32   q15, q12, %e16[0]   \n"
                    "vmla.f32   q7, q9, %e16[1]     \n"

                    "vld1.f32   {d16-d19}, [%4]     \n"
                    "add        %4, #16             \n"
                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"
                    "vext.32    q12, q8, q9, #3     \n"

                    "vmla.f32   q7, q8, %f16[0]     \n"
                    "vmla.f32   q13, q10, %f16[1]   \n"

                    "pld        [%5, #256]          \n"

                    "vmla.f32   q14, q11, %e17[0]   \n"
                    "vmla.f32   q15, q12, %e17[1]   \n"
                    "vmla.f32   q7, q9, %f17[0]     \n"

                    "vld1.f32   {d16-d19}, [%5]     \n"
                    "add        %5, #16             \n"
                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"
                    "vext.32    q12, q8, q9, #3     \n"

                    "vmla.f32   q7, q8, %f17[1]     \n"
                    "vmla.f32   q13, q10, %e18[0]   \n"

                    "pld        [%6, #256]          \n"

                    "vmla.f32   q14, q11, %e18[1]   \n"
                    "vmla.f32   q15, q12, %f18[0]   \n"
                    "vmla.f32   q7, q9, %f18[1]     \n"

                    "vld1.f32   {d16-d19}, [%6]     \n"
                    "add        %6, #16             \n"
                    "vext.32    q10, q8, q9, #1     \n"
                    "vext.32    q11, q8, q9, #2     \n"
                    "vext.32    q12, q8, q9, #3     \n"

                    "vmla.f32   q7, q8, %e19[0]     \n"
                    "vmla.f32   q13, q10, %e19[1]   \n"
                    "vmla.f32   q14, q11, %f19[0]   \n"
                    "vmla.f32   q15, q12, %f19[1]   \n"
                    "vmla.f32   q7, q9, %e20[0]     \n"

                    "vadd.f32   q14, q14, q15       \n"
                    "vadd.f32   q7, q7, q13         \n"
//                     "veor       q15, q15            \n"// _sum3 = 0;

                    "pld        [%2, #256]          \n"

                    "vadd.f32   q7, q7, q14         \n"

                    "vld1.f32   {d16-d19}, [%2]     \n"// _r00 = vld1q_f32(r0+j);
                    "add        %2, #16             \n"

                    "vst1.f32   {d14-d15}, [%1]!    \n"

                    "pld        [%1, #128]          \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    "sub        %2, #16             \n"
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

*/
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
    printf("the %d of cc is : %f\n", i, *(out+i));
    printf("-----------------the %d of dd is : %f\n", i, *(out2+i));
  }
  
/*  
  
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
  
  for (int i=0; i<outw*outh; i++)
  {
    printf("the %d of cc is : %f\n", i, *(out3+i));
    printf("-----------------the %d of dd is : %f\n", i, *(out4+i));
  }

*/

  delete []out;
  delete []out2;
  //delete []out3;
  //delete []out4;
    
  return 0;
}

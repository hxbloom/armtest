#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>

int main()
{
  // Test the assembly part only
  // Input is set to 36, so that outw=outh=9, leave remaining to be 1.
  const int w = 36;
  const int h = 36;
  const int outw = w/4;
  const int outh = h/4;
  assert(outw==9);
  assert(outh==9);
  
  const int tailstep = w - 4*outw + w*3;
  
  int nn = outw >> 2;
  int nn2 = outw >> 2;
  int remain = outw - (nn << 2);
  int remain2 = outw - (nn2 << 2);  
  
  float c0[w*h] __attribute__((aligned(16))) ;
  float d0[w*h] __attribute__((aligned(16))) ;
  for (int i=0; i<w*h; i++)
  {
    c0[i] = i+1;
    d0[i] = i+1;
  }
  
  // Kernels
  float ker1[16] __attribute__((aligned(16)))  = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}; // Test kernel for main part
  float ker2[16] __attribute__((aligned(16)))  = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2}; // Test kernel for remaining
  float* kernel0 = ker1;
  
  const float *r0 = c0;
  const float *r1 = c0 + w;
  const float *r2 = c0 + w * 2;
  const float *r3 = c0 + w * 3;
  
  const float *r02 = c0;
  const float *r12 = c0 + w;
  const float *r22 = c0 + w * 2;
  const float *r32 = c0 + w * 3;
  float *outptr  = new float[outw*outh]();
  float *outptr2 = new float[outw*outh]();

            // k0 - k3 is to test the sequential code. Will not appear in arm intrinsics/assembly code.
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 4;
            const float* k2 = kernel0 + 8;
            const float* k3 = kernel0 + 12;

            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k4567 = vld1q_f32(kernel0+4);
            float32x4_t _k891011 = vld1q_f32(kernel0+8);
            float32x4_t _k12131415 = vld1q_f32(kernel0+12);


// Line 74
// Case: Computer two channels at a time
// Intrinsic

            for (int i = 0; i < outh; i++)
            {
                nn2 = outw >> 2;
                remain2 = outw - (nn2 << 2);
                assert(remain2==1);
                
                for (; nn2>0; nn2--)
                {
                    float32x4_t _r00 = vld1q_f32(r02);
                    float32x4_t _r10 = vld1q_f32(r12);
                    float32x4_t _r20 = vld1q_f32(r22);
                    float32x4_t _r30 = vld1q_f32(r32);

                    float32x4_t _r01 = vld1q_f32(r02 + 4);
                    float32x4_t _r11 = vld1q_f32(r12 + 4);
                    float32x4_t _r21 = vld1q_f32(r22 + 4);
                    float32x4_t _r31 = vld1q_f32(r32 + 4);

                    float32x4_t _r02 = vld1q_f32(r02 + 8);
                    float32x4_t _r12 = vld1q_f32(r12 + 8);
                    float32x4_t _r22 = vld1q_f32(r22 + 8);
                    float32x4_t _r32 = vld1q_f32(r32 + 8);

                    float32x4_t _r03 = vld1q_f32(r02 + 12);
                    float32x4_t _r13 = vld1q_f32(r12 + 12);
                    float32x4_t _r23 = vld1q_f32(r22 + 12);
                    float32x4_t _r33 = vld1q_f32(r32 + 12);

                    float32x4_t _sum0 = vmulq_f32(_r00, _k0123);
                    float32x4_t _sum1 = vmulq_f32(_r01, _k0123);
                    float32x4_t _sum2 = vmulq_f32(_r02, _k0123);
                    float32x4_t _sum3 = vmulq_f32(_r03, _k0123);

                    _sum0 = vfmaq_f32(_sum0, _r10, _k4567);
                    _sum1 = vfmaq_f32(_sum1, _r11, _k4567);
                    _sum2 = vfmaq_f32(_sum2, _r12, _k4567);
                    _sum3 = vfmaq_f32(_sum3, _r13, _k4567);

                    _sum0 = vfmaq_f32(_sum0, _r20, _k891011);
                    _sum1 = vfmaq_f32(_sum1, _r21, _k891011);
                    _sum2 = vfmaq_f32(_sum2, _r22, _k891011);
                    _sum3 = vfmaq_f32(_sum3, _r23, _k891011);

                    _sum0 = vfmaq_f32(_sum0, _r30, _k12131415);
                    _sum1 = vfmaq_f32(_sum1, _r31, _k12131415);
                    _sum2 = vfmaq_f32(_sum2, _r32, _k12131415);
                    _sum3 = vfmaq_f32(_sum3, _r33, _k12131415);

                    float32x4_t _s01 = vpaddq_f32(_sum0, _sum1);
                    float32x4_t _s23 = vpaddq_f32(_sum2, _sum3);
                    float32x4_t _sum = vpaddq_f32(_s01, _s23);

                    float32x4_t _outp = vld1q_f32(outptr2);

                    _outp = vaddq_f32(_outp, _sum);

                    vst1q_f32(outptr2, _outp);

                    r02 += 16;
                    r12 += 16;
                    r22 += 16;
                    r32 += 16;
                    outptr2 += 4;
                }
                
                for (; remain2>0; remain2--)
                {
                    float32x4_t _r0 = vld1q_f32(r02);
                    float32x4_t _r1 = vld1q_f32(r12);
                    float32x4_t _r2 = vld1q_f32(r22);
                    float32x4_t _r3 = vld1q_f32(r32);

                    float32x4_t _sum = vmulq_f32(_r0, _k0123);
                    _sum = vmlaq_f32(_sum, _r1, _k4567);
                    _sum = vmlaq_f32(_sum, _r2, _k891011);
                    _sum = vmlaq_f32(_sum, _r3, _k12131415);

                    *outptr2 += vaddvq_f32(_sum);

                    r02 += 4;
                    r12 += 4;
                    r22 += 4;
                    r32 += 4;
                    outptr2++;
                }
                
                r02 += tailstep;
                r12 += tailstep;
                r22 += tailstep;
                r32 += tailstep;
            }

// Assembly
   
            for (int i = 0; i < outh; i++)
            {
                nn = outw >> 2;
                remain = outw - (nn << 2);
                assert(remain==1);

                if (nn > 0)
                {
                asm volatile(
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "0:                                        \n"

                    "prfm       pldl1keep, [%2, #512]          \n"
                    "prfm       pldl1keep, [%3, #512]          \n"

                    "ld1        {v7.4s}, [%1]                  \n" // v7 = outptr

                    "ld1        {v8.4s}, [%2], #16             \n"// v8  = r0
                    "ld1        {v9.4s}, [%3], #16             \n"// v9  = r1

                    "prfm       pldl1keep, [%4, #512]          \n"
                    "prfm       pldl1keep, [%5, #512]          \n"

                    "fmul       v12.4s, v8.4s, %12.4s          \n"
                    "fmul       v13.4s, v9.4s, %13.4s          \n"

                    "ld1        {v10.4s}, [%4], #16            \n"// v10 = r2
                    "ld1        {v11.4s}, [%5], #16            \n"// v11 = r3

                    "fmla       v12.4s, v10.4s, %14.4s         \n"
                    "fmla       v13.4s, v11.4s, %15.4s         \n"

                    "fadd       v5.4s, v12.4s, v13.4s          \n"

                    "ld1        {v8.4s}, [%2], #16             \n"// v8  = r0
                    "ld1        {v9.4s}, [%3], #16             \n"// v9  = r1

                    "fmul       v12.4s, v8.4s, %12.4s          \n"
                    "fmul       v13.4s, v9.4s, %13.4s          \n"

                    "ld1        {v10.4s}, [%4], #16            \n"// v10 = r2
                    "ld1        {v11.4s}, [%5], #16            \n"// v11 = r3
                    
                    "fmla       v12.4s, v10.4s, %14.4s         \n"
                    "fmla       v13.4s, v11.4s, %15.4s         \n"

                    "fadd       v6.4s, v12.4s, v13.4s          \n"

                    "ld1        {v8.4s}, [%2], #16             \n"// v8  = r0
                    "ld1        {v9.4s}, [%3], #16             \n"// v9  = r1

                    "fmul       v12.4s, v8.4s, %12.4s          \n"
                    "fmul       v13.4s, v9.4s, %13.4s          \n"

                    "ld1        {v10.4s}, [%4], #16            \n"// v10 = r2
                    "ld1        {v11.4s}, [%5], #16            \n"// v11 = r3

                    "fmla       v12.4s, v10.4s, %14.4s         \n"
                    "fmla       v13.4s, v11.4s, %15.4s         \n"

                    "fadd       v14.4s, v12.4s, v13.4s         \n"
                    "faddp      v5.4s, v5.4s, v6.4s            \n"  // Move to here to enhance ILP

                    "ld1        {v8.4s}, [%2], #16             \n"// v8  = r0
                    "ld1        {v9.4s}, [%3], #16             \n"// v9  = r1

                    "fmul       v12.4s, v8.4s, %12.4s          \n"
                    "fmul       v13.4s, v9.4s, %13.4s          \n"

                    "ld1        {v10.4s}, [%4], #16            \n"// v10 = r2
                    "ld1        {v11.4s}, [%5], #16            \n"// v11 = r3

                    "fmla       v12.4s, v10.4s, %14.4s         \n"
                    "fmla       v13.4s, v11.4s, %15.4s         \n"

                    "fadd       v15.4s, v12.4s, v13.4s         \n"

//                  "faddp      v5.4s ,  v5.4s,  v6.4s         \n"  // Move this line upward.
                    "faddp      v14.4s, v14.4s, v15.4s         \n"
                    "faddp      v5.4s ,  v5.4s, v14.4s         \n"
            
                    "fadd       v7.4s, v7.4s, v5.4s            \n"

                    "st1        {v7.4s}, [%1], #16             \n"

                    "prfm       pldl1keep, [%1, #128]          \n"

                    "subs       %w0, %w0, #1                   \n"
                    "bne        0b                             \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr),     // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3)          // %5
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "w"(_k0123),      // %12
                      "w"(_k4567),      // %13
                      "w"(_k891011),    // %14
                      "w"(_k12131415)   // %15
                    : "cc", "memory", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }

                for (; remain>0; remain--)
                {
                    float sum = 0.f;

                    asm volatile(
                        "ld1        {v8.4s}, [%0], #16             \n"// v8  = r0
                        "ld1        {v9.4s}, [%1], #16             \n"// v9  = r1

                        "fmul       v12.4s, v8.4s, %9.4s           \n"
                        "fmul       v13.4s, v9.4s, %10.4s          \n"

                        "ld1        {v10.4s}, [%2], #16            \n"// v10 = r2
                        "ld1        {v11.4s}, [%3], #16            \n"// v11 = r3

                        "fmla       v12.4s, v10.4s, %11.4s         \n"
                        "fmla       v13.4s, v11.4s, %12.4s         \n"

                        "fadd       v5.4s, v12.4s, v13.4s          \n"            
                        "faddp      v5.4s, v5.4s, v5.4s            \n"  
                        "faddp      s5, v5.2s                      \n" 
                        "fmov       %w4, s5                        \n"
                        : "=r"(r0),         // %0
                          "=r"(r1),         // %1
                          "=r"(r2),         // %2
                          "=r"(r3),         // %3
                          "=r"(sum)         // %4
                        : "0"(r0),
                          "1"(r1),
                          "2"(r2),
                          "3"(r3),
                          "w"(_k0123),      // %9
                          "w"(_k4567),      // %10
                          "w"(_k891011),    // %11
                          "w"(_k12131415)   // %12
                        : "cc", "memory", "v5", "v6", "v8", "v9", "v10", "v11", "v12", "v13"
                    );

                    *outptr += sum;
                    
                    outptr++;
                }
                
                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
            }

  outptr -= outw*outh;
  outptr2 -= outw*outh;
  
  for (int i=0; i<outw*outh; i++)
  {
    printf("the %d of cc is : %f\n", i, *(outptr+i));
    printf("-----------------the %d of dd is : %f\n", i, *(outptr2+i));
  }

  delete []outptr;
  delete []outptr2;
  
  return 0;
}

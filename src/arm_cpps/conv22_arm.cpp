#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>

int main()
{
  // Test the assembly part only
  // Set to 16 to let 'int remain = outw - (nn << 2);' in the original code to be 0
  const int w = 9;
  const int h = 9;
  const int outw = w-1;
  const int outh = h-1;
  assert(outw==8);
  assert(outh==8);
  
  const int tailstep = w - 2*outw + w;
  
  int nn = outw >> 2;
  int nn2 = outw >> 2;
  
  float c0[w*h];
  float d0[w*h];
  float e0[w*h];
  for (int i=0; i<w*h; i++)
  {
    c0[i] = i+1;
    d0[i] = i+1;
    e0[i] = i+1;
  }
  
  // Kernels
  float k1[8] __attribute__((aligned(16)))  = {1,1,1,1,2,2,2,2};
  float k2[4] __attribute__((aligned(16)))  = {3,3,3,3};
  float* kernel0 = k1;
  float* kernel1 = kernel0 + 4;
  
  const float *r00 = c0;
  const float *r01 = c0 + w;
  const float *r10 = d0;
  const float *r11 = d0 + w;
  
  const float *r002 = c0;
  const float *r012 = c0 + w;
  const float *r102 = d0;
  const float *r112 = d0 + w;
  float *outptr  = new float[outw*outh]();
  float *outptr2 = new float[outw*outh]();

            float32x4_t _k0 = vld1q_f32(kernel0);
            float32x4_t _k1 = vld1q_f32(kernel1);


// Line 74
// Case: Computer two channels at a time
// Intrinsic

            for (int i = 0; i < outh; i++)
            {
                nn2 = outw >> 2;
                for (; nn2>0; nn2--)
                {
                    float32x4_t _r000 = vld1q_f32(r002);
                    float32x4_t _r010 = vld1q_f32(r012);
                    float32x4_t _r001 = vld1q_f32(r002 + 1);
                    float32x4_t _r011 = vld1q_f32(r012 + 1);

                    float32x4_t _r100 = vld1q_f32(r102);
                    float32x4_t _r110 = vld1q_f32(r112);
                    float32x4_t _r101 = vld1q_f32(r102 + 1);
                    float32x4_t _r111 = vld1q_f32(r112 + 1);

                    float32x4_t _sum = vld1q_f32(outptr2);

                    _sum = vmlaq_lane_f32(_sum, _r000, vget_low_f32(_k0), 0);
                    _sum = vmlaq_lane_f32(_sum, _r001, vget_low_f32(_k0), 1);
                    _sum = vmlaq_lane_f32(_sum, _r010, vget_high_f32(_k0), 0);
                    _sum = vmlaq_lane_f32(_sum, _r011, vget_high_f32(_k0), 1);

                    _sum = vmlaq_lane_f32(_sum, _r100, vget_low_f32(_k1), 0);
                    _sum = vmlaq_lane_f32(_sum, _r101, vget_low_f32(_k1), 1);
                    _sum = vmlaq_lane_f32(_sum, _r110, vget_high_f32(_k1), 0);
                    _sum = vmlaq_lane_f32(_sum, _r111, vget_high_f32(_k1), 1);

                    vst1q_f32(outptr2, _sum);

                    r002 += 4;
                    r012 += 4;
                    r102 += 4;
                    r112 += 4;
                    outptr2 += 4;
                }

                r002 += 1;
                r012 += 1;
                r102 += 1;
                r112 += 1;
                
                }

// Assembly

            for (int i = 0; i < outh; i++)
            {
                nn = outw >> 2;
                asm volatile(
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1], #16             \n"
                    "prfm       pldl1keep, [%2, #128]          \n"
                    "ld1        {v2.4s}, [%2], #16             \n"
                    "prfm       pldl1keep, [%3, #128]          \n"
                    "ld1        {v12.4s}, [%3], #16            \n"
                    "prfm       pldl1keep, [%4, #128]          \n"
                    "ld1        {v14.4s}, [%4], #16            \n"

                    "0:                                        \n"
                    "prfm       pldl1keep, [%5, #128]          \n"
                    "ld1        {v9.4s}, [%5]                  \n"

                    "fmul       v8.4s, v0.4s, %12.s[0]         \n"
                    "fmla       v9.4s, v2.4s, %12.s[2]         \n"
                    
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v1.4s}, [%1], #16             \n"

                    "prfm       pldl1keep, [%2, #128]          \n"
                    "ld1        {v3.4s}, [%2], #16             \n"

                    "ext        v10.16b, v0.16b, v1.16b, #4    \n"
                    "ext        v11.16b, v2.16b, v3.16b, #4    \n"

                    "fmla       v8.4s, v12.4s, %13.s[0]        \n"
                    "fmla       v9.4s, v14.4s, %13.s[2]        \n"

                    "prfm       pldl1keep, [%3, #128]          \n"
                    "ld1        {v13.4s}, [%3], #16            \n"

                    "prfm       pldl1keep, [%4, #128]          \n"
                    "ld1        {v15.4s}, [%4], #16            \n"

                    "fmla       v8.4s, v10.4s, %12.s[1]        \n"
                    "fmla       v9.4s, v11.4s, %12.s[3]        \n"

                    "ext        v10.16b, v12.16b, v13.16b, #4  \n"
                    "ext        v11.16b, v14.16b, v15.16b, #4  \n"

                    "fmla       v8.4s, v10.4s, %13.s[1]        \n"
                    "fmla       v9.4s, v11.4s, %13.s[3]        \n"

                    "orr        v0.16b, v1.16b, v1.16b         \n"
                    "orr        v2.16b, v3.16b, v3.16b         \n"

                    "fadd       v8.4s, v8.4s, v9.4s            \n"

                    "orr        v12.16b, v13.16b, v13.16b      \n"
                    "orr        v14.16b, v15.16b, v15.16b      \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v8.4s}, [%5], #16             \n"
                    "bne        0b                             \n"
                    "sub        %1, %1, #16                    \n"
                    "sub        %2, %2, #16                    \n"
                    "sub        %3, %3, #16                    \n"
                    "sub        %4, %4, #16                    \n"
                    : "=r"(nn),     // %0
                      "=r"(r00),    // %1
                      "=r"(r01),    // %2
                      "=r"(r10),    // %3
                      "=r"(r11),    // %4
                      "=r"(outptr)  // %5
                    : "0"(nn),
                      "1"(r00),
                      "2"(r01),
                      "3"(r10),
                      "4"(r11),
                      "5"(outptr),
                      "w"(_k0),     // %12
                      "w"(_k1)      // %13
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );

                r00 += 1;
                r01 += 1;
                r10 += 1;
                r11 += 1;
                
                }


  outptr -= outw*outh;
  outptr2 -= outw*outh;
  


// Line 266
// Case: Computer one channel at a time (the last channel if needed)
// Intrinsic
  kernel0 = k2;
  const float *r0 = e0;
  const float *r1 = e0 + w;
  const float *r02 = e0;
  const float *r12 = e0 + w;
  
            _k0 = vdupq_n_f32(kernel0[0]);
            _k1 = vdupq_n_f32(kernel0[1]);
            float32x4_t _k2 = vdupq_n_f32(kernel0[2]);
            float32x4_t _k3 = vdupq_n_f32(kernel0[3]);

            for (int i = 0; i < outh; i++)
            {
                nn2 = outw >> 2;
                for (; nn2>0; nn2--)
                {
                    float32x4_t _r00 = vld1q_f32(r02);
                    float32x4_t _r10 = vld1q_f32(r12);
                    float32x4_t _r01 = vld1q_f32(r02 + 1);
                    float32x4_t _r11 = vld1q_f32(r12 + 1);

                    float32x4_t _sum = vld1q_f32(outptr2);
                    float32x4_t _sum2;

                    _sum = vmlaq_f32(_sum, _r00, _k0);
                    _sum2 = vmulq_f32(_r01, _k1);
                    _sum = vmlaq_f32(_sum, _r10, _k2);
                    _sum2 = vmlaq_f32(_sum2, _r11, _k3);

                    _sum = vaddq_f32(_sum, _sum2);

                    vst1q_f32(outptr2, _sum);

                    r02 += 4;
                    r12 += 4;
                    outptr2 += 4;
                }
                r02 += 1;
                r12 += 1;
            }

  
// Assembly
            for (int i = 0; i < outh; i++)
            {
                nn = outw >> 2;
                asm volatile(
                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v0.4s}, [%1], #16             \n"
                    "prfm       pldl1keep, [%2, #128]          \n"
                    "ld1        {v2.4s}, [%2], #16             \n"

                    "0:                                        \n"
                    "prfm       pldl1keep, [%3, #128]          \n"
                    "ld1        {v9.4s}, [%3]                  \n"

                    "fmul       v8.4s, v0.4s, %8.4s           \n"
                    "fmla       v9.4s, v2.4s, %10.4s          \n"

                    "prfm       pldl1keep, [%1, #128]          \n"
                    "ld1        {v1.4s}, [%1], #16             \n"
                    "ext        v10.16b, v0.16b, v1.16b, #4    \n"

                    "fmla       v8.4s, v10.4s, %9.4s           \n"

                    "prfm       pldl1keep, [%2, #128]          \n"
                    "ld1        {v3.4s}, [%2], #16             \n"
                    "ext        v11.16b, v2.16b, v3.16b, #4    \n"

                    "fmla       v9.4s, v11.4s, %11.4s          \n"

                    "orr        v0.16b, v1.16b, v1.16b         \n"
                    "fadd       v8.4s, v8.4s, v9.4s            \n"
                    "orr        v2.16b, v3.16b, v3.16b         \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v8.4s}, [%3], #16             \n"
                    "bne        0b                             \n"
                    "sub        %1, %1, #16                    \n"
                    "sub        %2, %2, #16                    \n"
                    : "=r"(nn),     // %0
                      "=r"(r0),     // %1
                      "=r"(r1),     // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(r0),
                      "2"(r1),
                      "3"(outptr),
                      "w"(_k0),     // %8
                      "w"(_k1),     // %9
                      "w"(_k2),     // %10
                      "w"(_k3)      // %11
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11"
                );
            
                r0 += 1;
                r1 += 1;
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

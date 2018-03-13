#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>

int main()
{
  // Test the assembly part only
  // Input is set to 10\11 respectively, so that outw=outh=9, let outh to be odd number.
  // Test is for 1 group only. (group is not considered)
  const int w = 10;
  const int h = 11;
  const int outw = w-2;
  const int outh = h-2;
  assert(outw==8);
  assert(outh==9);
  
  int nn = outw >> 2;
  int nn2 = outw >> 2;
  
  float c0[w*h] __attribute__((aligned(16))) ;
  float d0[w*h] __attribute__((aligned(16))) ;
  for (int i=0; i<w*h; i++)
  {
    c0[i] = i+1;
    d0[i] = i+1;
  }
  
  // Kernels, k0 for 
  float ker1[9] __attribute__((aligned(16)))  = {1,1,1,1,1,1,1,1,1}; // Test kernel for main part
  float ker2[9] __attribute__((aligned(16)))  = {2,2,2,2,2,2,2,2,2}; // Test kernel for remaining
  float* kernel0 = ker1;
  
  const float bias0 = 10000.f;
  
  const float *r0 = c0;
  const float *r1 = c0 + w;
  const float *r2 = c0 + w * 2;
  const float *r3 = c0 + w * 3;
  
  const float *r02 = c0;
  const float *r12 = c0 + w;
  const float *r22 = c0 + w * 2;
  const float *r32 = c0 + w * 3;
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


// Assembly

        i = 0;   
        for (; i+1 < outh; i+=2)
        {
            int nn = outw >> 2;

            if (nn > 0)
            {
            asm volatile(
                "prfm       pldl1keep, [%3, #192]          \n"
                "ld1        {v9.4s, v10.4s}, [%3]          \n" //r0
                "add        %3, %3, #16                  \n"

                "ext        v11.16b, v9.16b, v10.16b, #4   \n"
                "ext        v12.16b, v9.16b, v10.16b, #8   \n"

                "0:                                        \n"

                "fmul       v7.4s, v9.4s, %14.s[0]         \n"
                
                "and        v13.16b, %17.16b, %17.16b      \n" // v13 = _bias0
                "fmul       v6.4s,  v11.4s, %14.s[1]       \n"
                "fmla       v13.4s, v12.4s, %14.s[2]       \n"

                "prfm       pldl1keep, [%4, #192]          \n"
                "ld1        {v9.4s, v10.4s}, [%4]          \n"
                "add        %4, %4, #16                  \n"

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
                "add        %5, %5, #16                  \n"

                "fmla       v7.4s, v9.4s, %16.s[0]       \n"

                "ext        v11.16b, v9.16b, v10.16b, #4   \n"
                "ext        v12.16b, v9.16b, v10.16b, #8   \n"

                "fmla       v6.4s,  v11.4s, %16.s[1]       \n"
                "fmla       v13.4s, v12.4s, %16.s[2]       \n"

                "fmla       v8.4s,   v9.4s, %15.s[0]       \n"
                "fmla       v14.4s, v11.4s, %15.s[1]       \n"
                "fmla       v15.4s, v12.4s, %15.s[2]       \n"

                "prfm       pldl1keep, [%6, #192]          \n"
                "ld1        {v9.4s, v10.4s}, [%6]          \n"
                "add        %6, %6, #16                  \n"

                "fmla       v8.4s, v9.4s, %16.s[0]         \n"

                "ext        v11.16b, v9.16b, v10.16b, #4   \n"
                "ext        v12.16b, v9.16b, v10.16b, #8   \n"

                "fmla       v14.4s, v11.4s, %16.s[1]       \n"
                "fmla       v15.4s, v12.4s, %16.s[2]       \n"

                "fadd       v7.4s, v7.4s, v6.4s            \n"

                "prfm       pldl1keep, [%3, #192]          \n"
                "ld1        {v9.4s, v10.4s}, [%3]          \n" //ro, for next loop

                "fadd       v8.4s, v8.4s, v14.4s            \n"
                "fadd       v7.4s, v7.4s, v13.4s            \n"
                "fadd       v8.4s, v8.4s, v15.4s            \n"

                "ext        v11.16b, v9.16b, v10.16b, #4   \n" // for next loop
                "ext        v12.16b, v9.16b, v10.16b, #8   \n" // for next loop

                "add        %3, %3, #16                  \n"
                
                "st1        {v7.4s}, [%1], #16             \n"
                "st1        {v8.4s}, [%2], #16             \n"

                "subs       %w0, %w0, #1                   \n"
                "bne        0b                             \n"

                "sub        %3, %3, #16                  \n"
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



/*
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

*/



  outptr -= outw*(outh-1);
  outptrb -= outw*(outh-1);

//  outptr -= outw*outh;
//  outptrb -= outw*outh;
  
  for (int i=0; i<outw*outh; i++)
  {
    printf("the %d of cc is : %f\n", i, *(outptr+i));
    printf("-----------------the %d of dd is : %f\n", i, *(outptrb+i));
  }

  delete []out;
  delete []out2;
  
  return 0;
}

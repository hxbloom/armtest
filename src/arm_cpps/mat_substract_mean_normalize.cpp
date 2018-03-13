#include <arm_neon.h>
#include <stdio.h>


int main()
{
  int nn = 5;
  int nn2 = 5;
  float mean = 5.f;
  float norm = 10.f;
  
  float c[20] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float d[20] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float e[20] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float f[20] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float g[20] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float h[20] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
  float *cc = c;
  float *dd = d;
  float *ee = e;
  float *ff = f;
  float *gg = g;
  float *hh = h;

  float *ptr = (float*)cc;
  float *ptr2 = (float*)dd;

// Line 29    if (mean_vals && !norm_vals)
// substract mean only
            float32x4_t _mean = vdupq_n_f32(mean);
            for (; nn2>0; nn2--)
            {
                float32x4_t _ptr2 = vld1q_f32(ptr2);
                _ptr2 = vsubq_f32(_ptr2, _mean);
                vst1q_f32(ptr2, _ptr2);
                ptr2 += 4;
            }

// substract mean only: assembly
            asm volatile(
                "dup        v1.4s, %w4            \n"
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "ld1        {v0.4s}, [%1]         \n"
                "fsub       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%1], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(mean)     // %4
                : "cc", "memory", "v0", "v1"
            );



// Line 84    else if (!mean_vals && norm_vals)
// normalize only
  nn = 5;
  nn2 = 5;
  ptr = (float*)ee;
  ptr2 = (float*)ff;
  
            float32x4_t _norm = vdupq_n_f32(norm);
            for (; nn2>0; nn2--)
            {
                float32x4_t _ptr2 = vld1q_f32(ptr2);
                _ptr2 = vmulq_f32(_ptr2, _norm);
                vst1q_f32(ptr2, _ptr2);
                ptr2 += 4;
            }

// normalize only: assembly
            asm volatile(
                "dup        v1.4s, %w4            \n"
                "0:                             \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "ld1        {v0.4s}, [%1]         \n"
                "fmul       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%1], #16    \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(norm)     // %4
                : "cc", "memory", "v0", "v1"
            );


// Line 139        else if (mean_vals && norm_vals)
// substract mean and normalize
  nn = 5;
  nn2 = 5;
  ptr = (float*)gg;
  ptr2 = (float*)hh;

            _mean = vdupq_n_f32(mean);
            _norm = vdupq_n_f32(norm);
            for (; nn2>0; nn2--)
            {
                float32x4_t _ptr2 = vld1q_f32(ptr2);
                _ptr2 = vsubq_f32(_ptr2, _mean);
                _ptr2 = vmulq_f32(_ptr2, _norm);
                vst1q_f32(ptr2, _ptr2);
                ptr2 += 4;
            }

// substract mean and normalize:assembly
            asm volatile(
                "dup        v1.4s, %w4            \n"
                "dup        v2.4s, %w5            \n"
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "ld1        {v0.4s}, [%1]         \n"
                "fsub       v0.4s, v0.4s, v1.4s   \n"
                "fmul       v0.4s, v0.4s, v2.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%1], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(mean),    // %4
                  "r"(norm)     // %5
                : "cc", "memory", "v0", "v1", "v2"
            );  
  
  
  
  
  for (int i=0; i<20; i++)
  {
    printf("the %d of cc is : %f\n", i, *(gg+i));
    printf("-----the %d of dd is : %f\n", i, *(hh+i));
  }

  return 0;
}

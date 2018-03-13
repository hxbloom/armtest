#include <arm_neon.h>
#include <stdio.h>


int main()
{
  int nn = 5;
  int nn2 = 5;
  float a = 1000;
  float b = 2;
  
  float c0[20] = {-5,-4,-3,-2,-1,0,1,2,3,4,4,3,2,1,0,-1,-2,-3,-4,-5};
  float d0[20] = {-5,-4,-3,-2,-1,0,1,2,3,4,4,3,2,1,0,-1,-2,-3,-4,-5};
  
  // For Product
  float *cc = c0;
  float *dd = d0;
  
  float *ptr = (float*)cc;
  float *ptr2 = (float*)dd;
  

// ****** Product
// Line 46
        float32x4_t _a = vdupq_n_f32(a);
        float32x4_t _b = vdupq_n_f32(b);
        for (; nn2>0; nn2--)
        {
            float32x4_t _p = vld1q_f32(ptr2);
            float32x4_t _outp = _a;
            _outp = vfmaq_f32(_outp, _p, _b);
            vst1q_f32(ptr2, _outp);

            ptr2 += 4;
        }


 
// The aarch64 assembly here contains prefetch

        asm volatile(
            "dup        v1.4s, %w4            \n"
            "dup        v2.4s, %w5            \n"
            "0:                               \n"
            "prfm       pldl1keep, [%1, #128] \n"
            "ld1        {v0.4s}, [%1]         \n"
            "orr        v3.16b, v1.16b, v1.16b\n"
            "fmla       v3.4s, v0.4s, v2.4s   \n"
            "subs       %w0, %w0, #1          \n"
            "st1        {v3.4s}, [%1], #16    \n"
            "bne        0b                    \n"
            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            : "0"(nn),
              "1"(ptr),
              "r"(a),       // %4
              "r"(b)        // %5
            : "cc", "memory", "v0", "v1", "v2", "v3"
        );

/*
        asm volatile(
            "vdup.f32   q1, %4              \n"
            "vdup.f32   q2, %5              \n"
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld1.f32   {d0-d1}, [%1 :128]  \n"
            "vorr.32    q3, q1, q1          \n"
            "vmla.f32   q3, q0, q2          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d6-d7}, [%1 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            : "0"(nn),
              "1"(ptr),
              "r"(a),       // %4
              "r"(b)        // %5
            : "cc", "memory", "q0", "q1", "q2", "q3"
        );
*/




  
  for (int i=0; i<20; i++)
  {
    printf("the %d of cc is : %f\n", i, *(cc+i));
    printf("-----------the %d of dd is : %f\n", i, *(dd+i));
  }

  return 0;
}

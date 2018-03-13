#include <arm_neon.h>
#include <stdio.h>


int main()
{
  int nn = 5;
  int nn2 = 5;
  
  float c0[20] = {-5,-4,-3,-2,-1,0,1,2,3,4,4,3,2,1,0,-1,-2,-3,-4,-5};
  
  // For Product
  float *cc = c0;
  float *dd = c0;
  
  float *ptr = (float*)cc;
  float *ptr2 = (float*)dd;
  

// ****** Product
// Line 46
        for (; nn2>0; nn2--)
        {
            float32x4_t _p = vld1q_f32(ptr2);
            _p = vabsq_f32(_p);
            vst1q_f32(ptr2, _p);

            ptr2 += 4;
        }

// The aarch64 assembly here contains prefetch
        asm volatile(
            "0:                               \n"
            "prfm       pldl1keep, [%1, #128] \n"
            "ld1        {v0.4s}, [%1]         \n"
            "fabs       v0.4s, v0.4s          \n"
            "subs       %w0, %w0, #1          \n"
            "st1        {v0.4s}, [%1], #16    \n"
            "bne        0b                    \n"
            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            : "0"(nn),
              "1"(ptr)
            : "cc", "memory", "v0"
        );
        

/*
        asm volatile(
            "0:                             \n"
// Optional: Add prefetch here
//            "pld        [%1, #128]          \n"
            "vld1.f32   {d0-d1}, [%1]       \n"
            "vabs.f32   q0, q0              \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d1}, [%1]!      \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            : "0"(nn),
              "1"(ptr)
            : "cc", "memory", "q0"
        );
*/
  
  for (int i=0; i<20; i++)
  {
    printf("the %d of cc is : %f\n", i, *(cc+i));
    printf("-----------the %d of dd is : %f\n", i, *(dd+i));
  }

  return 0;
}

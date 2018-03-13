#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>

int main()
{
  // Test the assembly part only
  // Input is set to 36, so that outw=outh=9, leave remaining to be 1.
  const int w = 4;
  const int h = 4;
  int nn = 1;
  
  float c0[w*h] __attribute__((aligned(16))) ;
  float d0[w*h] __attribute__((aligned(16))) ;
  for (int i=0; i<w*h; i++)
  {
    c0[i] = i+1;
    d0[i] = i+1;
  }
  
  const float *r0 = c0;
  const float *r1 = c0 + w;
  const float *r2 = c0 + w*2;
  const float *r3 = c0 + w*3;
  float *outptr  = new float[8]();
  
  float sum = 0.f;
  
          asm volatile(
            "0:                               \n"
            "ld1        {v0.4s}, [%1], #16    \n"
            "ld1        {v1.4s}, [%2], #16    \n"
            
            "fadd       v0.4s, v0.4s, v1.4s   \n"            
            "faddp      v0.4s, v0.4s, v0.4s   \n"  
            "faddp      s0, v0.2s             \n" 
                        
//            "ld1        {v2.4s}, [%3], #16    \n"
//            "ld1        {v3.4s}, [%4], #16    \n"
            
//            "faddp      v0.4s, v0.4s, v1.4s   \n"
//            "faddp      v2.4s, v2.4s, v3.4s   \n"
//            "faddp      v0.4s, v0.4s, v2.4s   \n"
            
            "subs       %w0, %w0, #1          \n"
//            "st1        {v0.4s}, [%5], #16    \n"
            "fmov       %w5, s0           \n"
            "bne        0b                    \n"
            : "=r"(nn),     // %0
              "=r"(r0),     // %1
              "=r"(r1),     // %2
              "=r"(r2),     // %3
              "=r"(r3),     // %4
              "=r"(sum)  // %5             
            : "0"(nn),
              "1"(r0),
              "2"(r1),
              "3"(r2),
              "4"(r3),
              "5"(sum) 
            : "cc", "memory", "v0", "v1", "v2", "v3"
        );
        
  outptr -= 4;

  printf("the sum is : %f .\n", sum);

  for (int i=0; i<8; i++)
  {
    //printf("the %d of cc is : %f\n", i, *(outptr+i));
  }

  return 0;
}

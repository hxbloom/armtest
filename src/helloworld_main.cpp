#include <iostream>
#include <arm_neon.h>

void add_float_c(float* dst, float* src1, float* src2, int count)
{
     int i;
     for (i = 0; i < count; i++)
         dst[i] = src1[i] + src2[i];
}


void add_float_neon1(float* dst, float* src1, float* src2, int count)
{
     int i;
     for (i = 0; i < count; i += 4)
     {
         float32x4_t in1, in2, out;
         in1 = vld1q_f32(src1);
         src1 += 4;
         in2 = vld1q_f32(src2);
         src2 += 4;
         out = vaddq_f32(in1, in2);
         vst1q_f32(dst, out);
         dst += 4;                     
     }
}

#if __ARM_NEON
#if __aarch64__
void add_float_neon3(float* dst, float* src1, float* src2, int count)
{

 asm volatile (
 "1: \n"
 "ld1 {v0.4s}, [%[src1]], #16 \n"
 "ld1 {v1.4s}, [%[src2]], #16 \n"
 "fadd v0.4s, v0.4s, v1.4s \n"
 "subs %w[count], %w[count], #4 \n"
 "st1 {v0.4s}, [%[dst]], #16 \n"
 "bgt 1b \n"
 : [dst] "+r" (dst)
 : [src1] "r" (src1), [src2] "r" (src2), [count] "r" (count)
 : "memory", "v0", "v1"
 );

 /* 
  asm volatile (
    "1: \n"
    "ld1 {v0.4s}, [%1], #16 \n"
    "ld1 {v1.4s}, [%2], #16 \n"
    "fadd v0.4s, v0.4s, v1.4s \n"
    "subs %w3, %w3, #4 \n"
    "st1 {v0.4s}, [%0], #16 \n"
    "bgt 1b \n"
    : "=r"(dst), "=r"(src1), "=r"(src2), "=r"(count)
    : "0"(dst), "1"(src1), "2"(src2), "3"(count)
    : "memory", "v0", "v1"
 );
 */
}
#else
void add_float_neon3(float* dst, float* src1, float* src2, int count)
{
 asm volatile (
 "1: \n"
 "vld1.32 {q0}, [%[src1]]! \n"
 "vld1.32 {q1}, [%[src2]]! \n"
 "vadd.f32 q0, q0, q1 \n"
 "subs %[count], %[count], #4 \n"
 "vst1.32 {q0}, [%[dst]]! \n"
 "bgt 1b \n"
 : [dst] "+r" (dst)
 : [src1] "r" (src1), [src2] "r" (src2), [count] "r" (count)
 : "memory", "q0", "q1"
 );
}

#endif // __aarch64__
#endif // __ARM_NEON

int main() {

#ifdef __clang__
    std::cout <<"----------　Compiled with clang -----------" << std::endl;
#else
    std::cout <<"----------　Compiled with gcc -----------" << std::endl;
#endif

#if __ARM_NEON
#if __aarch64__
    std::cout <<"----------　With aarch64 NEON-----------" << std::endl;
#elif __arm__
    std::cout <<"----------　Wtih arm 32 NEON-----------" << std::endl;
#endif // __aarch64__
#endif // __ARM_NEON

    std::cout <<"Hello World!" << std::endl;
    
    // Only available after c++11
    //float* test1 = new float[4] {1.f, 2.f, 3.f,   4.f};
    //float* test2 = new float[4] {5.f, 6.f, 7.f, 100.f};
    float* test1 = new float[4];
    float* test2 = new float[4];
    test1[0] = 1.f; test1[1] = 2.f; test1[2] = 3.f; test1[3] =   4.f;
    test2[0] = 5.f; test2[1] = 6.f; test2[2] = 3.f; test2[3] = 100.f;
    
    float* sum1 = new float[4]();
    float* sum2 = new float[4]();
    float* sum3 = new float[4]();
    float* sum4 = new float[4]();

    add_float_c(sum1, test1, test2, 4);
    add_float_neon1(sum2, test1, test2, 4);
    //int* ct = new int(4);
    add_float_neon3(sum4, test1, test2, 4);
    
    for (int i=0; i<4; i++)
    {
        std::cout << "sum1[" << i << "] = " << sum1[i] << std::endl;
        std::cout << "sum2[" << i << "] = " << sum2[i] << std::endl;
        //std::cout << "sum3[" << i << "] = " << sum3[i] << std::endl;
        std::cout << "sum4[" << i << "] = " << sum4[i] << std::endl;
    }
    
    delete[] test1;
    delete[] test2;
    delete[] sum1;
    delete[] sum2;
    delete[] sum3;
    delete[] sum4;
    
    return 0;
}

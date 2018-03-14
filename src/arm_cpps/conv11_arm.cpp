#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>


#define ENABLE_BIAS 1

#ifdef ENABLE_BIAS
#include <algorithm>
#endif

int main()
{
  // *************************************************************
  // ********                                           **********
  // ********   Part 1: Convolution 1*1 with stride 1   **********
  // ********                                           **********
  // *************************************************************
  
  
  // Line 994:
  //
  //    for (int pp=0; pp<nn_outch; pp++)
  //        int p = pp * 4;
  //        for (; q<inch; q++)
  //
  // Computer 4*output channel, with the remaining 1*input channel
  
  int w = 16;
  int h = 1;
  int outw = w;
  int outh = h*4;
  assert(outw==16);
  assert(outh==4);
  
  int nn = outw >> 3;
  int nn2 = outw >> 3;
  
  float c0[w*h] __attribute__((aligned(16))) ;
  for (int i=0; i<w*h; i++)
  {
    c0[i] = i+1;
  }
  
  // Kernels, k1 for 
  float ker1[4] __attribute__((aligned(16)))  = {1,2,3,4};  // 4*output channel, with the remaining 1*input channel
  float* kernel0 = ker1;
  
  float k0 = kernel0[0];
  float k1 = kernel0[1];
  float k2 = kernel0[2];
  float k3 = kernel0[3];
  
  float *r0 = c0;
  float *r0b = c0;
  
  float *out  = new float[outw*outh]();
  float *out2 = new float[outw*outh]();
  
  float* outptr0 = out;
  float* outptr1 = out + outw;
  float* outptr2 = out + outw * 2;
  float* outptr3 = out + outw * 3;
  float* outptr0b = out2;
  float* outptr1b = out2 + outw;
  float* outptr2b = out2 + outw * 2;
  float* outptr3b = out2 + outw * 3;

#ifdef ENABLE_BIAS
  // Define and Fill the bias
  float bias0 = 10000.f;
  float bias1 = 20000.f;
  float bias2 = 30000.f;
  float bias3 = 40000.f;
    
  std::fill_n(outptr0, outw, bias0); 
  std::fill_n(outptr1, outw, bias1);
  std::fill_n(outptr2, outw, bias2); 
  std::fill_n(outptr3, outw, bias3);
  std::fill_n(outptr0b, outw, bias0); 
  std::fill_n(outptr1b, outw, bias1);
  std::fill_n(outptr2b, outw, bias2); 
  std::fill_n(outptr3b, outw, bias3);
#endif


            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);

// Line 1030
// 4*output channel, with the remaining 1*input channel
// Intrinsic
#if __aarch64__

            for (; nn2>0; nn2--)
            {
                float32x4_t _p = vld1q_f32(r0b);
                float32x4_t _pn = vld1q_f32(r0b+4);

                float32x4_t _out0p = vld1q_f32(outptr0b);
                float32x4_t _out0pn = vld1q_f32(outptr0b+4);

                float32x4_t _out1p = vld1q_f32(outptr1b);
                float32x4_t _out1pn = vld1q_f32(outptr1b+4);

                float32x4_t _out2p = vld1q_f32(outptr2b);
                float32x4_t _out2pn = vld1q_f32(outptr2b+4);

                float32x4_t _out3p = vld1q_f32(outptr3b);
                float32x4_t _out3pn = vld1q_f32(outptr3b+4);

                _out0p = vfmaq_f32(_out0p, _p, _k0);    
                _out0pn = vfmaq_f32(_out0pn, _pn, _k0);

                _out1p = vfmaq_f32(_out1p, _p, _k1);
                _out1pn = vfmaq_f32(_out1pn, _pn, _k1);

                _out2p = vfmaq_f32(_out2p, _p, _k2);
                _out2pn = vfmaq_f32(_out2pn, _pn, _k2);

                _out3p = vfmaq_f32(_out3p, _p, _k3);
                _out3pn = vfmaq_f32(_out3pn, _pn, _k3);

                vst1q_f32(outptr0b, _out0p);
                vst1q_f32(outptr0b+4, _out0pn);

                vst1q_f32(outptr1b, _out1p);
                vst1q_f32(outptr1b+4, _out1pn);

                vst1q_f32(outptr2b, _out2p);
                vst1q_f32(outptr2b+4, _out2pn);

                vst1q_f32(outptr3b, _out3p);
                vst1q_f32(outptr3b+4, _out3pn);

                r0b += 8;
                outptr0b += 8;
                outptr1b += 8;
                outptr2b += 8;
                outptr3b += 8;
            }

#endif // __aarch64__

// Line 1080
// 4*output channel, with the remaining 1*input channel
// Assembly
#if __aarch64__

            if (nn > 0)
            {
            asm volatile(
                "prfm       pldl1keep, [%5, #256]          \n"
                "ld1        {v6.4s, v7.4s}, [%5], #32      \n"
                "0:                                        \n"
                "prfm       pldl1keep, [%1, #256]          \n"
                "ld1        {v8.4s, v9.4s}, [%1]           \n"
                "fmla       v8.4s, v6.4s, %12.4s           \n"
                "fmla       v9.4s, v7.4s, %12.4s           \n"
                
                "prfm       pldl1keep, [%2, #256]          \n"
                "ld1        {v10.4s, v11.4s}, [%2]         \n"
                "fmla       v10.4s, v6.4s, %13.4s          \n"
                "fmla       v11.4s, v7.4s, %13.4s          \n"

                "st1        {v8.4s, v9.4s}, [%1], #32      \n"

                "prfm       pldl1keep, [%3, #256]          \n"
                "ld1        {v12.4s, v13.4s}, [%3]         \n"
                "fmla       v12.4s, v6.4s, %14.4s          \n"
                "fmla       v13.4s, v7.4s, %14.4s          \n"

                "st1        {v10.4s, v11.4s}, [%2], #32    \n"

                "prfm       pldl1keep, [%4, #256]          \n"
                "ld1        {v14.4s, v15.4s}, [%4]         \n"
                "fmla       v14.4s, v6.4s, %15.4s          \n"
                "fmla       v15.4s, v7.4s, %15.4s          \n"

                "st1        {v12.4s, v13.4s}, [%3], #32    \n"

                "prfm       pldl1keep, [%5, #256]          \n"
                "ld1        {v6.4s, v7.4s}, [%5], #32      \n"
                "subs       %w0, %w0, #1                   \n"
                "st1        {v14.4s, v15.4s}, [%4], #32    \n"
                "bne        0b                             \n"
                "sub        %5, %5, #32                    \n"
                : "=r"(nn),     // %0
                  "=r"(outptr0),// %1
                  "=r"(outptr1),// %2
                  "=r"(outptr2),// %3
                  "=r"(outptr3),// %4
                  "=r"(r0)      // %5
                : "0"(nn),
                  "1"(outptr0),
                  "2"(outptr1),
                  "3"(outptr2),
                  "4"(outptr3),
                  "5"(r0),
                  "w"(_k0),     // %12
                  "w"(_k1),     // %13
                  "w"(_k2),     // %14
                  "w"(_k3)      // %15
                : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            );
            }

#endif // __aarch64__

  for (int i=0; i<outw*outh; i++)
  {
//    printf("the %d of cc is : %f\n", i, *(out+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out2+i));
  }















  // Line 1172:
  //
  //    for (int p=remain_outch_start; p<outch; p++)
  //        for (; q+3<inch; q+=4)
  //
  // Computer 1*output channel(remaining), with the 4*input channel

  w = 16;
  h = 4;
  outw = w;
  outh = h/4;
  assert(outw==16);
  assert(outh==1);

  nn = outw >> 3;
  nn2 = outw >> 3;

  float d0[w*h] __attribute__((aligned(16))) ;
  for (int i=0; i<w*h; i++)
  {
    d0[i] = i+1;
  }

  r0 = d0;
  float *r1 = d0 + w;
  float *r2 = d0 + w * 2;
  float *r3 = d0 + w * 3;
  r0b = d0;
  float *r1b = d0 + w;
  float *r2b = d0 + w * 2;
  float *r3b = d0 + w * 3;

  float *out3 = new float[outw*outh]();
  float *out4 = new float[outw*outh]();

  float* outptr = out3;
  float* outptrb = out4;

  // Kernels, k2 for 
  float ker2[4] __attribute__((aligned(16)))  = {1,1,1,1};  // 1*output channel(remaining), with the 4*input channel
  kernel0 = ker2;

  k0 = kernel0[0];
  k1 = kernel0[1];
  k2 = kernel0[2];
  k3 = kernel0[3];

#ifdef ENABLE_BIAS
  // Define and Fill the bias
  bias0 = 50000.f;
    
  std::fill_n(outptr, outw*outh, bias0); 
  std::fill_n(outptrb, outw*outh, bias0); 
#endif

            _k0 = vdupq_n_f32(k0);
            _k1 = vdupq_n_f32(k1);
            _k2 = vdupq_n_f32(k2);
            _k3 = vdupq_n_f32(k3);

// Line 1205
// 1*output channel(remaining), with the 4*input channel
// Intrinsic
#if __aarch64__

            for (; nn2>0; nn2--)
            {
                float32x4_t _p = vld1q_f32(r0b);
                float32x4_t _pn = vld1q_f32(r0b+4);

                float32x4_t _outp = vld1q_f32(outptrb);
                float32x4_t _outpn = vld1q_f32(outptrb+4);

                _outp = vfmaq_f32(_outp, _p, _k0);
                _outpn = vfmaq_f32(_outpn, _pn, _k0);

                float32x4_t _p1 = vld1q_f32(r1b);
                float32x4_t _p1n = vld1q_f32(r1b+4);

                _outp = vfmaq_f32(_outp, _p1, _k1);
                _outpn = vfmaq_f32(_outpn, _p1n, _k1);

                float32x4_t _p2 = vld1q_f32(r2b);
                float32x4_t _p2n = vld1q_f32(r2b+4);

                _outp = vfmaq_f32(_outp, _p2, _k2);
                _outpn = vfmaq_f32(_outpn, _p2n, _k2);

                float32x4_t _p3 = vld1q_f32(r3b);
                float32x4_t _p3n = vld1q_f32(r3b+4);

                _outp = vfmaq_f32(_outp, _p3, _k3);
                _outpn = vfmaq_f32(_outpn, _p3n, _k3);

                vst1q_f32(outptrb, _outp);
                vst1q_f32(outptrb+4, _outpn);

                r0b += 8;
                r1b += 8;
                r2b += 8;
                r3b += 8;
                outptrb += 8;
            }

#endif // __aarch64__


// Line 1245
// 1*output channel(remaining), with the 4*input channel
// Assembly

#if __aarch64__

            if (nn > 0)
            {
            asm volatile(
                "prfm       pldl1keep, [%2, #256]          \n"
                "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                "0:                                        \n"
                "prfm       pldl1keep, [%1, #256]          \n"
                "ld1        {v0.4s, v1.4s}, [%1]           \n"
                "fmla       v0.4s, v2.4s, %12.4s           \n"
                "fmla       v1.4s, v3.4s, %12.4s           \n"

                "prfm       pldl1keep, [%3, #256]          \n"
                "ld1        {v2.4s, v3.4s}, [%3], #32      \n"
                "fmla       v0.4s, v2.4s, %13.4s           \n"
                "fmla       v1.4s, v3.4s, %13.4s           \n"

                "prfm       pldl1keep, [%4, #256]          \n"
                "ld1        {v2.4s, v3.4s}, [%4], #32      \n"
                "fmla       v0.4s, v2.4s, %14.4s           \n"
                "fmla       v1.4s, v3.4s, %14.4s           \n"

                "prfm       pldl1keep, [%5, #256]          \n"
                "ld1        {v2.4s, v3.4s}, [%5], #32      \n"
                "fmla       v0.4s, v2.4s, %15.4s           \n"
                "fmla       v1.4s, v3.4s, %15.4s           \n"

                "prfm       pldl1keep, [%2, #256]          \n"
                "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                "subs       %w0, %w0, #1                   \n"
                "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                "bne        0b                             \n"
                "sub        %2, %2, #32                    \n"
                : "=r"(nn),     // %0
                  "=r"(outptr), // %1
                  "=r"(r0),     // %2
                  "=r"(r1),     // %3
                  "=r"(r2),     // %4
                  "=r"(r3)      // %5
                : "0"(nn),
                  "1"(outptr),
                  "2"(r0),
                  "3"(r1),
                  "4"(r2),
                  "5"(r3),
                  "w"(_k0),     // %12
                  "w"(_k1),     // %13
                  "w"(_k2),     // %14
                  "w"(_k3)      // %15
                : "cc", "memory", "v0", "v1", "v2", "v3"
            );
            }

#endif // __aarch64__

  for (int i=0; i<outw*outh; i++)
  {
//    printf("the %d of cc is : %f\n", i, *(out3+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out4+i));
  }















  // Line 1311:
  //
  //    for (int p=remain_outch_start; p<outch; p++)
  //        for (; q<inch; q++)
  //
  // Computer 1*output channel(remaining), with the 1*input channel(remaining)

  w = 16;
  h = 1;
  outw = w;
  outh = h;
  assert(outw==16);
  assert(outh==1);

  nn = w >> 3;
  nn2 = w >> 3;

  float e0[w*h] __attribute__((aligned(16))) ;
  for (int i=0; i<w*h; i++)
  {
    e0[i] = i+1;
  }

  r0 = e0;
  r0b = e0;

  float *out5 = new float[outw*outh]();
  float *out6 = new float[outw*outh]();

  outptr = out5;
  outptrb = out6;

  // Kernels, k3 for 
  float ker3[1] __attribute__((aligned(16)))  = {10};  // 1*output channel(remaining), with the 1*input channel
  kernel0 = ker3;

  k0 = kernel0[0];

#ifdef ENABLE_BIAS
  // Define and Fill the bias
  bias0 = 60000.f;
    
  std::fill_n(outptr, outw*outh, bias0); 
  std::fill_n(outptrb, outw*outh, bias0); 
#endif

            _k0 = vdupq_n_f32(k0);

// Line 1334
// 1*output channel(remaining), with the 1*input channel(remaining)
// Intrinsic
#if __aarch64__

            for (; nn2>0; nn2--)
            {
                float32x4_t _p = vld1q_f32(r0b);
                float32x4_t _outp = vld1q_f32(outptrb);

                float32x4_t _pn = vld1q_f32(r0b+4);
                float32x4_t _outpn = vld1q_f32(outptrb+4);

                _outp = vfmaq_f32(_outp, _p, _k0);
                _outpn = vfmaq_f32(_outpn, _pn, _k0);

                vst1q_f32(outptrb, _outp);
                vst1q_f32(outptrb+4, _outpn);

                r0b += 8;
                outptrb += 8;
            }

#endif // __aarch64__
// Line 1354
// 1*output channel(remaining), with the 1*input channel(remaining)
// Assembly
#if __aarch64__
            if (nn > 0)
            {
            asm volatile(
                "prfm       pldl1keep, [%2, #256]          \n"
                "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                "0:                                        \n"
                "prfm       pldl1keep, [%1, #256]          \n"
                "ld1        {v0.4s, v1.4s}, [%1]           \n"
                "fmla       v0.4s, v2.4s, %6.4s            \n"
                "fmla       v1.4s, v3.4s, %6.4s            \n"
                "prfm       pldl1keep, [%2, #256]          \n"
                "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                "subs       %w0, %w0, #1                   \n"
                "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                "bne        0b                             \n"
                "sub        %2, %2, #32                    \n"
                : "=r"(nn),     // %0
                  "=r"(outptr), // %1
                  "=r"(r0)      // %2
                : "0"(nn),
                  "1"(outptr),
                  "2"(r0),
                  "w"(_k0)      // %6
                : "cc", "memory", "v0", "v1", "v2", "v3"
            );
            }
#endif // __aarch64__

  for (int i=0; i<outw*outh; i++)
  {
//    printf("the %d of cc is : %f\n", i, *(out5+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out6+i));
  }


  delete []out;
  delete []out2;
  delete []out3;
  delete []out4;
  delete []out5;
  delete []out6;
  

















  // *************************************************************
  // ********                                           **********
  // ********   Part 2: Convolution 1*1 with stride 2   **********
  // ********                                           **********
  // *************************************************************


  // Line 1413:
  //
  //    for (int pp=0; pp<nn_outch; pp++)
  //        int p = pp * 4;
  //        for (; q+3<inch; q+=4)
  //
  // Computer 4*output channel, with the 4*input channel

  const int tailstep = w - 2*outw + w;
  
  w = 16;
  h = 4;
  outw = w;
  outh = h;
  assert(outw==16);
  assert(outh==4);

  nn = outw >> 3;
  nn2 = outw >> 3;

  float f0[2*w*4] __attribute__((aligned(16))) ;
  for (int i=0; i<w*2; i++)
  {
    f0[i] = i+1;
    f0[2*w*1+i] = i+1;
    f0[2*w*2+i] = i+1;
    f0[2*w*3+i] = i+1;
  }

  r0 = f0;
  r1 = f0 + w * 2;
  r2 = f0 + w * 2 * 2;
  r3 = f0 + w * 2 * 3;
  r0b = f0;
  r1b = f0 + w * 2;
  r2b = f0 + w * 2 * 2;
  r3b = f0 + w * 2 * 3;

  float *out7 = new float[outw*outh]();
  float *out8 = new float[outw*outh]();

  outptr0 = out7;
  outptr1 = out7 + outw;
  outptr2 = out7 + outw * 2;
  outptr3 = out7 + outw * 3;
  outptr0b = out8;
  outptr1b = out8 + outw;
  outptr2b = out8 + outw * 2;
  outptr3b = out8 + outw * 3;


  // Kernels, k4 for 
  float ker4[16] __attribute__((aligned(16)))  = {1,1,1,1, \
                                                  2,2,2,2, \
                                                  3,3,3,3, \
                                                  4,4,4,4};  // Computer 4*output channel, with the 4*input channel

//  float ker4[4] __attribute__((aligned(16)))  = {1,1,1,1};
  kernel0 = ker4;

  float *k0_f = kernel0;
  float *k1_f = kernel0 + 4;
  float *k2_f = kernel0 + 8;
  float *k3_f = kernel0 + 12;

//  k0 = kernel0[0];
//  k1 = kernel0[1];
//  k2 = kernel0[2];
//  k3 = kernel0[3];


#ifdef ENABLE_BIAS
  // Define and Fill the bias
  bias0 = 40000.f;
  bias1 = 50000.f;
  bias2 = 60000.f;
  bias3 = 70000.f;
    
  std::fill_n(outptr0, outw, bias0); 
  std::fill_n(outptr1, outw, bias1);
  std::fill_n(outptr2, outw, bias2); 
  std::fill_n(outptr3, outw, bias3);
  std::fill_n(outptr0b, outw, bias0); 
  std::fill_n(outptr1b, outw, bias1);
  std::fill_n(outptr2b, outw, bias2); 
  std::fill_n(outptr3b, outw, bias3);
#endif

                _k0 = vld1q_f32(k0_f);
                _k1 = vld1q_f32(k1_f);
                _k2 = vld1q_f32(k2_f);
                _k3 = vld1q_f32(k3_f);
//            _k0 = vdupq_n_f32(k0);
//            _k1 = vdupq_n_f32(k1);
//            _k2 = vdupq_n_f32(k2);
//            _k3 = vdupq_n_f32(k3);


// Line 1473
// 4*output channel, with the 4*input channel
// Intrinsic
#if __aarch64__
//            for (int i = 0; i < outh; i++)
            for (int i = 0; i < 1; i++)
            {
                nn2 = outw >> 3;

                for (; nn2>0; nn2--)
                {
  
                    float32x4x2_t _px2 = vld2q_f32(r0b);
                    float32x4_t _p = _px2.val[0];
                    float32x4x2_t _pnx2 = vld2q_f32(r0b+8);
                    float32x4_t _pn = _pnx2.val[0];

                    float32x4_t _out0p = vld1q_f32(outptr0b);
                    float32x4_t _out0pn = vld1q_f32(outptr0b+4);

                    float32x4_t _out1p = vld1q_f32(outptr1b);
                    float32x4_t _out1pn = vld1q_f32(outptr1b+4);

                    float32x4_t _out2p = vld1q_f32(outptr2b);
                    float32x4_t _out2pn = vld1q_f32(outptr2b+4);

                    float32x4_t _out3p = vld1q_f32(outptr3b);
                    float32x4_t _out3pn = vld1q_f32(outptr3b+4);

                    _out0p = vfmaq_laneq_f32(_out0p, _p, _k0, 0);
                    _out0pn = vfmaq_laneq_f32(_out0pn, _pn, _k0, 0);

                    _out1p = vfmaq_laneq_f32(_out1p, _p, _k1, 0);
                    _out1pn = vfmaq_laneq_f32(_out1pn, _pn, _k1, 0);

                    _out2p = vfmaq_laneq_f32(_out2p, _p, _k2, 0);
                    _out2pn = vfmaq_laneq_f32(_out2pn, _pn, _k2, 0);

                    _out3p = vfmaq_laneq_f32(_out3p, _p, _k3, 0);
                    _out3pn = vfmaq_laneq_f32(_out3pn, _pn, _k3, 0);

                    float32x4x2_t _p1x2 = vld2q_f32(r1b);
                    float32x4_t _p1 = _p1x2.val[0];
                    float32x4x2_t _p1nx2 = vld2q_f32(r1b+8);
                    float32x4_t _p1n = _p1nx2.val[0];

                    _out0p = vfmaq_laneq_f32(_out0p, _p1, _k0, 1);
                    _out0pn = vfmaq_laneq_f32(_out0pn, _p1n, _k0, 1);

                    _out1p = vfmaq_laneq_f32(_out1p, _p1, _k1, 1);
                    _out1pn = vfmaq_laneq_f32(_out1pn, _p1n, _k1, 1);

                    _out2p = vfmaq_laneq_f32(_out2p, _p1, _k2, 1);
                    _out2pn = vfmaq_laneq_f32(_out2pn, _p1n, _k2, 1);

                    _out3p = vfmaq_laneq_f32(_out3p, _p1, _k3, 1);
                    _out3pn = vfmaq_laneq_f32(_out3pn, _p1n, _k3, 1);

                    float32x4x2_t _p2x2 = vld2q_f32(r2b);
                    float32x4_t _p2 = _p2x2.val[0];
                    float32x4x2_t _p2nx2 = vld2q_f32(r2b+8);
                    float32x4_t _p2n = _p2nx2.val[0];

                    _out0p = vfmaq_laneq_f32(_out0p, _p2, _k0, 2);
                    _out0pn = vfmaq_laneq_f32(_out0pn, _p2n, _k0, 2);

                    _out1p = vfmaq_laneq_f32(_out1p, _p2, _k1, 2);
                    _out1pn = vfmaq_laneq_f32(_out1pn, _p2n, _k1, 2);

                    _out2p = vfmaq_laneq_f32(_out2p, _p2, _k2, 2);
                    _out2pn = vfmaq_laneq_f32(_out2pn, _p2n, _k2, 2);

                    _out3p = vfmaq_laneq_f32(_out3p, _p2, _k3, 2);
                    _out3pn = vfmaq_laneq_f32(_out3pn, _p2n, _k3, 2);

                    float32x4x2_t _p3x2 = vld2q_f32(r3b);
                    float32x4_t _p3 = _p3x2.val[0];
                    float32x4x2_t _p3nx2 = vld2q_f32(r3b+8);
                    float32x4_t _p3n = _p3nx2.val[0];

                    _out0p = vfmaq_laneq_f32(_out0p, _p3, _k0, 3);
                    _out0pn = vfmaq_laneq_f32(_out0pn, _p3n, _k0, 3);

                    _out1p = vfmaq_laneq_f32(_out1p, _p3, _k1, 3);
                    _out1pn = vfmaq_laneq_f32(_out1pn, _p3n, _k1, 3);

                    _out2p = vfmaq_laneq_f32(_out2p, _p3, _k2, 3);
                    _out2pn = vfmaq_laneq_f32(_out2pn, _p3n, _k2, 3);

                    _out3p = vfmaq_laneq_f32(_out3p, _p3, _k3, 3);
                    _out3pn = vfmaq_laneq_f32(_out3pn, _p3n, _k3, 3);

                    vst1q_f32(outptr0b, _out0p);
                    vst1q_f32(outptr0b+4, _out0pn);

                    vst1q_f32(outptr1b, _out1p);
                    vst1q_f32(outptr1b+4, _out1pn);

                    vst1q_f32(outptr2b, _out2p);
                    vst1q_f32(outptr2b+4, _out2pn);

                    vst1q_f32(outptr3b, _out3p);
                    vst1q_f32(outptr3b+4, _out3pn);

                    r0b += 16;
                    r1b += 16;
                    r2b += 16;
                    r3b += 16;
                    outptr0b += 8;
                    outptr1b += 8;
                    outptr2b += 8;
                    outptr3b += 8;
                }


                r0b += tailstep;
                r1b += tailstep;
                r2b += tailstep;
                r3b += tailstep;
            }
#endif // __aarch64__

// Line 1579
// 4*output channel, with the 4*input channel
// Assembly

#if __aarch64__
//            for (int i = 0; i < outh; i++)
            for (int i = 0; i < 1; i++)
            {
            
                nn = outw >> 3;
                if (nn > 0)
                {
                asm volatile(
                    "0:                                        \n"

                    "prfm       pldl1keep, [%5, #512]          \n"
                    "ld2        {v4.4s, v5.4s}, [%5], #32      \n"
                    "ld2        {v6.4s, v7.4s}, [%5], #32      \n"
                    "and        v5.16b, v6.16b, v6.16b         \n"// v4 v5

                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v8.4s, v9.4s}, [%1]           \n"

                    "fmla       v8.4s, v4.4s, %18.s[0]         \n"
                    "fmla       v9.4s, v5.4s, %18.s[0]         \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v10.4s, v11.4s}, [%2]         \n"

                    "fmla       v10.4s, v4.4s, %19.s[0]        \n"
                    "fmla       v11.4s, v5.4s, %19.s[0]        \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld1        {v12.4s, v13.4s}, [%3]         \n"

                    "fmla       v12.4s, v4.4s, %20.s[0]        \n"
                    "fmla       v13.4s, v5.4s, %20.s[0]        \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld1        {v14.4s, v15.4s}, [%4]         \n"

                    "prfm       pldl1keep, [%6, #512]          \n"
                    "ld2        {v6.4s, v7.4s}, [%6], #32      \n"

                    "fmla       v14.4s, v4.4s, %21.s[0]        \n"
                    "fmla       v15.4s, v5.4s, %21.s[0]        \n"

                    "ld2        {v4.4s, v5.4s}, [%6], #32      \n"
                    "and        v7.16b, v4.16b, v4.16b         \n"// v6 v7

                    "fmla       v8.4s, v6.4s, %18.s[1]         \n"
                    "fmla       v9.4s, v7.4s, %18.s[1]         \n"

                    "fmla       v10.4s, v6.4s, %19.s[1]        \n"
                    "fmla       v11.4s, v7.4s, %19.s[1]        \n"

                    "fmla       v12.4s, v6.4s, %20.s[1]        \n"
                    "fmla       v13.4s, v7.4s, %20.s[1]        \n"

                    "prfm       pldl1keep, [%7, #512]          \n"
                    "ld2        {v4.4s, v5.4s}, [%7], #32      \n"

                    "fmla       v14.4s, v6.4s, %21.s[1]        \n"
                    "fmla       v15.4s, v7.4s, %21.s[1]        \n"

                    "ld2        {v6.4s, v7.4s}, [%7], #32      \n"
                    "and        v5.16b, v6.16b, v6.16b         \n"// v4 v5

                    "fmla       v8.4s, v4.4s, %18.s[2]        \n"
                    "fmla       v9.4s, v5.4s, %18.s[2]        \n"

                    "fmla       v10.4s, v4.4s, %19.s[2]       \n"
                    "fmla       v11.4s, v5.4s, %19.s[2]       \n"

                    "fmla       v12.4s, v4.4s, %20.s[2]       \n"
                    "fmla       v13.4s, v5.4s, %20.s[2]       \n"

                    "prfm       pldl1keep, [%8, #512]          \n"
                    "ld2        {v6.4s, v7.4s}, [%8], #32      \n"

                    "fmla       v14.4s, v4.4s, %21.s[2]        \n"
                    "fmla       v15.4s, v5.4s, %21.s[2]        \n"

                    "ld2        {v4.4s, v5.4s}, [%8], #32      \n"
                    "and        v7.16b, v4.16b, v4.16b         \n"// v6 v7

                    "fmla       v8.4s, v6.4s, %18.s[3]        \n"
                    "fmla       v9.4s, v7.4s, %18.s[3]        \n"

                    "fmla       v10.4s, v6.4s, %19.s[3]       \n"
                    "fmla       v11.4s, v7.4s, %19.s[3]       \n"

                    "st1        {v8.4s, v9.4s}, [%1], #32     \n"

                    "fmla       v12.4s, v6.4s, %20.s[3]       \n"
                    "fmla       v13.4s, v7.4s, %20.s[3]       \n"

                    "st1        {v10.4s, v11.4s}, [%2], #32   \n"

                    "fmla       v14.4s, v6.4s, %21.s[3]       \n"
                    "fmla       v15.4s, v7.4s, %21.s[3]       \n"

                    "st1        {v12.4s, v13.4s}, [%3], #32   \n"

                    "subs       %w0, %w0, #1                  \n"
                    "st1        {v14.4s, v15.4s}, [%4], #32   \n"

                    "bne        0b                            \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr0),// %1
                      "=r"(outptr1),// %2
                      "=r"(outptr2),// %3
                      "=r"(outptr3),// %4
                      "=r"(r0),     // %5
                      "=r"(r1),     // %6
                      "=r"(r2),     // %7
                      "=r"(r3)      // %8
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(r3),
                      "w"(_k0),     // %18
                      "w"(_k1),     // %19
                      "w"(_k2),     // %20
                      "w"(_k3)      // %21
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
            }
#endif // __aarch64__



  for (int i=0; i<outw*outh; i++)
  {
//    printf("the %d of cc is : %f\n", i, *(out7+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out8+i));
  }














  // Line 1732:
  //
  //    for (int pp=0; pp<nn_outch; pp++)
  //        int p = pp * 4;
  //        for (; q<inch; q++)
  //
  // Computer 4*output channel, with the 1*input channel(remaining)

  w = 16;
  h = 1;
  outw = w;
  outh = h*4;
  assert(outw==16);
  assert(outh==4);

  nn = outw >> 3;
  nn2 = outw >> 3;

  float g0[2*w*4] __attribute__((aligned(16))) ;
  for (int i=0; i<w*2; i++)
  {
    g0[i] = i+1;
    g0[2*w*1+i] = i+1;
    g0[2*w*2+i] = i+1;
    g0[2*w*3+i] = i+1;
  }

  r0 = g0;
  r0b = g0;

  float *out9 = new float[outw*outh]();
  float *out10 = new float[outw*outh]();

  outptr0 = out9;
  outptr1 = out9 + outw;
  outptr2 = out9 + outw * 2;
  outptr3 = out9 + outw * 3;
  outptr0b = out10;
  outptr1b = out10 + outw;
  outptr2b = out10 + outw * 2;
  outptr3b = out10 + outw * 3;


  // Kernels, k5 for 
  float ker5[4] __attribute__((aligned(16)))  = {1,2,3,4};  // Computer 4*output channel, with the 1*input channel(remaining)
  kernel0 = ker5;

  k0 = kernel0[0];
  k1 = kernel0[1];
  k2 = kernel0[2];
  k3 = kernel0[3];


#ifdef ENABLE_BIAS
  // Define and Fill the bias
  bias0 = 60000.f;
  bias1 = 70000.f;
  bias2 = 80000.f;
  bias3 = 90000.f;
    
  std::fill_n(outptr0, outw, bias0); 
  std::fill_n(outptr1, outw, bias1);
  std::fill_n(outptr2, outw, bias2); 
  std::fill_n(outptr3, outw, bias3);
  std::fill_n(outptr0b, outw, bias0); 
  std::fill_n(outptr1b, outw, bias1);
  std::fill_n(outptr2b, outw, bias2); 
  std::fill_n(outptr3b, outw, bias3);
#endif

            _k0 = vdupq_n_f32(k0);
            _k1 = vdupq_n_f32(k1);
            _k2 = vdupq_n_f32(k2);
            _k3 = vdupq_n_f32(k3);



// Line 1772
// 4*output channel, with the 1*input channel(remaining)
// Intrinsic
#if __aarch64__
                for (; nn2>0; nn2--)
                {
                    float32x4x2_t _px2 = vld2q_f32(r0b);
                    float32x4_t _p = _px2.val[0];
                    float32x4x2_t _pnx2 = vld2q_f32(r0b+8);
                    float32x4_t _pn = _pnx2.val[0];

                    float32x4_t _out0p = vld1q_f32(outptr0b);
                    float32x4_t _out0pn = vld1q_f32(outptr0b+4);

                    float32x4_t _out1p = vld1q_f32(outptr1b);
                    float32x4_t _out1pn = vld1q_f32(outptr1b+4);

                    float32x4_t _out2p = vld1q_f32(outptr2b);
                    float32x4_t _out2pn = vld1q_f32(outptr2b+4);

                    float32x4_t _out3p = vld1q_f32(outptr3b);
                    float32x4_t _out3pn = vld1q_f32(outptr3b+4);

                    _out0p = vfmaq_f32(_out0p, _p, _k0);
                    _out0pn = vfmaq_f32(_out0pn, _pn, _k0);

                    _out1p = vfmaq_f32(_out1p, _p, _k1);
                    _out1pn = vfmaq_f32(_out1pn, _pn, _k1);

                    _out2p = vfmaq_f32(_out2p, _p, _k2);
                    _out2pn = vfmaq_f32(_out2pn, _pn, _k2);

                    _out3p = vfmaq_f32(_out3p, _p, _k3);
                    _out3pn = vfmaq_f32(_out3pn, _pn, _k3);

                    vst1q_f32(outptr0b, _out0p);
                    vst1q_f32(outptr0b+4, _out0pn);

                    vst1q_f32(outptr1b, _out1p);
                    vst1q_f32(outptr1b+4, _out1pn);

                    vst1q_f32(outptr2b, _out2p);
                    vst1q_f32(outptr2b+4, _out2pn);

                    vst1q_f32(outptr3b, _out3p);
                    vst1q_f32(outptr3b+4, _out3pn);

                    r0b += 16;
                    outptr0b += 8;
                    outptr1b += 8;
                    outptr2b += 8;
                    outptr3b += 8;
                }
                
#endif // __aarch64__



// Line 1822
// 4*output channel, with the 1*input channel(remaining)
// Assembly

#if __aarch64__

                if (nn > 0)
                {
                asm volatile(
                    "0:                                        \n"

                    "prfm       pldl1keep, [%5, #512]          \n"
                    "ld2        {v4.4s, v5.4s}, [%5], #32      \n"
                    "ld2        {v6.4s, v7.4s}, [%5], #32      \n"
                    "and        v5.16b, v6.16b, v6.16b         \n"
                    
                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v8.4s, v9.4s}, [%1]           \n"

                    "fmla       v8.4s, v4.4s, %12.4s           \n"
                    "fmla       v9.4s, v5.4s, %12.4s           \n"

                    "prfm       pldl1keep, [%2, #256]          \n"
                    "ld1        {v10.4s, v11.4s}, [%2]         \n"

                    "fmla       v10.4s, v4.4s, %13.4s          \n"
                    "fmla       v11.4s, v5.4s, %13.4s          \n"

                    "prfm       pldl1keep, [%3, #256]          \n"
                    "ld1        {v12.4s, v13.4s}, [%3]         \n"

                    "st1        {v8.4s, v9.4s}, [%1], #32      \n"

                    "fmla       v12.4s, v4.4s, %14.4s          \n"
                    "fmla       v13.4s, v5.4s, %14.4s          \n"

                    "prfm       pldl1keep, [%4, #256]          \n"
                    "ld1        {v14.4s, v15.4s}, [%4]         \n"

                    "st1        {v10.4s, v11.4s}, [%2], #32    \n"
                    
                    "fmla       v14.4s, v4.4s, %15.4s          \n"
                    "fmla       v15.4s, v5.4s, %15.4s          \n"

                    "st1        {v12.4s, v13.4s}, [%3], #32    \n"
                    "subs       %w0, %w0, #1                   \n"
                    
                    "st1        {v14.4s, v15.4s}, [%4], #32    \n"
                    "bne        0b                             \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr0),// %1
                      "=r"(outptr1),// %2
                      "=r"(outptr2),// %3
                      "=r"(outptr3),// %4
                      "=r"(r0)      // %5
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "w"(_k0),     // %12
                      "w"(_k1),     // %13
                      "w"(_k2),     // %14
                      "w"(_k3)      // %15
                    : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                );
                }

#endif // __aarch64__


  for (int i=0; i<outw*outh; i++)
  {
//    printf("the %d of cc is : %f\n", i, *(out9+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out10+i));
  }












  // Line 1919:
  //
  //    for (int p=remain_outch_start; p<outch; p++)
  //        for (; q+3<inch; q+=4)
  //
  // Computer 1*output channel(remaining), with the 4*input channel

  w = 16;
  h = 4;
  outw = w;
  outh = h/4;
  assert(outw==16);
  assert(outh==1);

  nn = outw >> 3;
  nn2 = outw >> 3;

  float h0[2*w*4] __attribute__((aligned(16))) ;
  for (int i=0; i<w*2; i++)
  {
    h0[i] = i+1;
    h0[2*w*1+i] = i+1;
    h0[2*w*2+i] = i+1;
    h0[2*w*3+i] = i+1;
  }

  r0 = h0;
  r1 = h0 + w * 2;
  r2 = h0 + w * 2 * 2;
  r3 = h0 + w * 2 * 3;
  r0b = h0;
  r1b = h0 + w * 2;
  r2b = h0 + w * 2 * 2;
  r3b = h0 + w * 2 * 3;

  float *out11 = new float[outw*outh]();
  float *out12 = new float[outw*outh]();

  outptr = out11;
  outptrb = out12;


  // Kernels, k6 for 
  float ker6[4] __attribute__((aligned(16)))  = {4,3,2,1}; // Computer 1*output channel(remaining), with the 4*input channel
  kernel0 = ker6;

  k0 = kernel0[0];
  k1 = kernel0[1];
  k2 = kernel0[2];
  k3 = kernel0[3];


#ifdef ENABLE_BIAS
  // Define and Fill the bias
  bias0 = 50000.f;
    
  std::fill_n(outptr, outw*outh, bias0); 
  std::fill_n(outptrb, outw*outh, bias0); 
#endif

            _k0 = vdupq_n_f32(k0);
            _k1 = vdupq_n_f32(k1);
            _k2 = vdupq_n_f32(k2);
            _k3 = vdupq_n_f32(k3);



// Line 1954
// 1*output channel(remaining), with the 4*input channel
// Intrinsic
#if __aarch64__
                for (; nn2>0; nn2--)
                {
                    float32x4x2_t _px2 = vld2q_f32(r0b);
                    float32x4_t _p = _px2.val[0];
                    float32x4_t _outp = vld1q_f32(outptrb);

                    float32x4x2_t _pnx2 = vld2q_f32(r0b+8);
                    float32x4_t _pn = _pnx2.val[0];
                    float32x4_t _outpn = vld1q_f32(outptrb+4);

                    _outp = vmlaq_f32(_outp, _p, _k0);
                    _outpn = vmlaq_f32(_outpn, _pn, _k0);

                    float32x4x2_t _p1x2 = vld2q_f32(r1b);
                    float32x4_t _p1 = _p1x2.val[0];
                    float32x4x2_t _p1nx2 = vld2q_f32(r1b+8);
                    float32x4_t _p1n = _p1nx2.val[0];

                    _outp = vmlaq_f32(_outp, _p1, _k1);
                    _outpn = vmlaq_f32(_outpn, _p1n, _k1);

                    float32x4x2_t _p2x2 = vld2q_f32(r2b);
                    float32x4_t _p2 = _p2x2.val[0];
                    float32x4x2_t _p2nx2 = vld2q_f32(r2b+8);
                    float32x4_t _p2n = _p2nx2.val[0];

                    _outp = vmlaq_f32(_outp, _p2, _k2);
                    _outpn = vmlaq_f32(_outpn, _p2n, _k2);

                    float32x4x2_t _p3x2 = vld2q_f32(r3b);
                    float32x4_t _p3 = _p3x2.val[0];
                    float32x4x2_t _p3nx2 = vld2q_f32(r3b+8);
                    float32x4_t _p3n = _p3nx2.val[0];

                    _outp = vmlaq_f32(_outp, _p3, _k3);
                    _outpn = vmlaq_f32(_outpn, _p3n, _k3);

                    vst1q_f32(outptrb, _outp);
                    vst1q_f32(outptrb+4, _outpn);

                    r0b += 16;
                    r1b += 16;
                    r2b += 16;
                    r3b += 16;
                    outptrb += 8;
                }


#endif //__aarch64__



// Line 2003
// 1*output channel(remaining), with the 4*input channel
// Assembly
#if __aarch64__

                if (nn > 0)
                {
                asm volatile(
                    "prfm       pldl1keep, [%2, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%2], #32      \n"
                    "0:                                        \n"

                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v0.4s, v1.4s}, [%1]           \n"
                    "fmla       v0.4s, v2.4s, %12.4s           \n"
                    "fmla       v1.4s, v8.4s, %12.4s           \n"

                    "prfm       pldl1keep, [%3, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%3], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%3], #32      \n"
                    "fmla       v0.4s, v2.4s, %13.4s           \n"
                    "fmla       v1.4s, v8.4s, %13.4s           \n"

                    "prfm       pldl1keep, [%4, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%4], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%4], #32      \n"
                    "fmla       v0.4s, v2.4s, %14.4s           \n"
                    "fmla       v1.4s, v8.4s, %14.4s           \n"

                    "prfm       pldl1keep, [%5, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%5], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%5], #32      \n"
                    "fmla       v0.4s, v2.4s, %15.4s           \n"
                    "fmla       v1.4s, v8.4s, %15.4s           \n"

                    "prfm       pldl1keep, [%2, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%2], #32      \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #64                    \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2),     // %4
                      "=r"(r3)      // %5
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "w"(_k0),     // %12
                      "w"(_k1),     // %13
                      "w"(_k2),     // %14
                      "w"(_k3)      // %15
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9"
                );
                }

#endif //__aarch64__



  for (int i=0; i<outw*outh; i++)
  {
//    printf("the %d of cc is : %f\n", i, *(out11+i));
//    printf("-----------------the %d of dd is : %f\n", i, *(out12+i));
  }

















  // Line 2079:
  //
  //    for (int p=remain_outch_start; p<outch; p++)
  //        for (; q<inch; q++)
  //
  // Computer 1*output channel(remaining), with the 1*input channel(remaining)

  w = 16;
  h = 1;
  outw = w;
  outh = h;
  assert(outw==16);
  assert(outh==1);

  nn = outw >> 3;
  nn2 = outw >> 3;

  float i0[2*w] __attribute__((aligned(16))) ;
  for (int i=0; i<w*2; i++)
  {
    i0[i] = i+1;
  }

  r0 = i0;
  r0b = i0;

  float *out13 = new float[outw*outh]();
  float *out14 = new float[outw*outh]();

  outptr = out13;
  outptrb = out14;


  // Kernels, k7 for 
  float ker7[1] __attribute__((aligned(16)))  = {100}; // Computer 1*output channel(remaining), with the 1*input channel
  kernel0 = ker7;

  k0 = kernel0[0];


#ifdef ENABLE_BIAS
  // Define and Fill the bias
  bias0 = 80000.f;
    
  std::fill_n(outptr, outw*outh, bias0); 
  std::fill_n(outptrb, outw*outh, bias0); 
#endif

            _k0 = vdupq_n_f32(k0);


// Line 2102
// 1*output channel(remaining), with the 1*input channel(remaining)
// Intrinsic
#if __aarch64__

                for (; nn2>0; nn2--)
                {
                    float32x4x2_t _px2 = vld2q_f32(r0b);
                    float32x4_t _p = _px2.val[0];
                    float32x4_t _outp = vld1q_f32(outptrb);

                    float32x4x2_t _pnx2 = vld2q_f32(r0b+8);
                    float32x4_t _pn = _pnx2.val[0];
                    float32x4_t _outpn = vld1q_f32(outptrb+4);

                    _outp = vmlaq_f32(_outp, _p, _k0);
                    _outpn = vmlaq_f32(_outpn, _pn, _k0);

                    vst1q_f32(outptrb, _outp);
                    vst1q_f32(outptrb+4, _outpn);

                    r0b += 16;
                    outptrb += 8;
                }

#endif //__aarch64__


// Line 2124
// 1*output channel(remaining), with the 1*input channel(remaining)
// Assembly
#if __aarch64__

                if (nn > 0)
                {
                asm volatile(
                    "prfm       pldl1keep, [%2, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%2], #32      \n"

                    "0:                                        \n"

                    "prfm       pldl1keep, [%1, #256]          \n"
                    "ld1        {v0.4s, v1.4s}, [%1]           \n"
                    "fmla       v0.4s, v2.4s, %6.4s            \n"
                    "fmla       v1.4s, v8.4s, %6.4s            \n"

                    "prfm       pldl1keep, [%2, #512]          \n"
                    "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                    "ld2        {v8.4s, v9.4s}, [%2], #32      \n"

                    "subs       %w0, %w0, #1                   \n"
                    "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                    "bne        0b                             \n"
                    "sub        %2, %2, #64                    \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0)      // %2
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "w"(_k0)      // %6
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9"
                );
                }



#endif //__aarch64__


  for (int i=0; i<outw*outh; i++)
  {
    printf("the %d of cc is : %f\n", i, *(out13+i));
    printf("-----------------the %d of dd is : %f\n", i, *(out14+i));
  }


  delete []out7;
  delete []out8;
  delete []out9;
  delete []out10;
  delete []out11;
  delete []out12;
  delete []out13;
  delete []out14;
  
  return 0;
}

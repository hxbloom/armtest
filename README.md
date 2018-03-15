# armtest


用 arm aarch64 NEON 汇编 重写了绝大部分现有 ncnn 中 aarch32 为汇编但 aarch64 依然为内部函数版本的代码。这个是 aarch64 汇编的测试集～



---

### 特性
* 不依赖原生ncnn，单独提取 NEON intrinsics 代码段并测试通过

* 使用 cmake，在 linux (Ubuntu 16.04) 下编译通过，编译方式类似 ncnn

* 目前有对应 aarch32 汇编的代码中，除了 /src/layer/mat_pixel.cpp 文件之外，其他所有对应 aarch64 版本都已经完成 （flag:但愿我没漏掉什么)

* 所有替换的代码段均配有单独的测试环境，使用 googletest 作为测试工具

* 绝大部分 aarch64 NEON 汇编基于对应的 aarch32 汇编逐行改写，降低出bug概率

* 唯一例外是 convolution7*7，因为卷积核较大原版 aarch32 里 SIMD 寄存器不够用，限制了性能。于是就写了一个新的 

* 所有测试集都在 ./tests/aarch64/ 下， ./src/下保留了改写时候用的带有 main() 的cpp文件




---

### clone 并编译该测试包

* 包含一个依赖库 : googletest ，所以 clone 的时候要加 `--recursive`

```bash
$ git clone --recursive https://github.com/hxbloom/armtest.git
```
或者使用

```bash
$ git submodule update --init --recursive
```

* 使用 cmake 编译：

需要下载 android-ndk ，方法核位置与 ncnn 相同

```bash
download android-ndk from http://developer.android.com/ndk/downloads/index.html
$ unzip android-ndk-r15c-linux-x86_64.zip
$ export ANDROID_NDK=<your-ndk-root-path>
```

编译对应的 aarch64 版本

```bash
$ cd <armtest-root-dir>
$ mkdir -p build-android-aarch64
$ cd build-android-aarch64
$ cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
     -DARM_ARCH="aarch64" ..
$ make -j4
```

编译后得到 aarch64_test ，传到手机等设备上运行即可 ^.^

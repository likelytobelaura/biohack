# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/likelytobelaura/work/mnist-et/device/kernel"
  "/home/likelytobelaura/work/mnist-et/device/build/kernel"
  "/home/likelytobelaura/work/mnist-et/device/build/mnist_kernel-prefix"
  "/home/likelytobelaura/work/mnist-et/device/build/mnist_kernel-prefix/tmp"
  "/home/likelytobelaura/work/mnist-et/device/build/mnist_kernel-prefix/src/mnist_kernel-stamp"
  "/home/likelytobelaura/work/mnist-et/device/build/mnist_kernel-prefix/src"
  "/home/likelytobelaura/work/mnist-et/device/build/mnist_kernel-prefix/src/mnist_kernel-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/likelytobelaura/work/mnist-et/device/build/mnist_kernel-prefix/src/mnist_kernel-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/likelytobelaura/work/mnist-et/device/build/mnist_kernel-prefix/src/mnist_kernel-stamp${cfgdir}") # cfgdir has leading slash
endif()

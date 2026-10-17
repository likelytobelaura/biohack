# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/likelytobelaura/osteo/aat/kernel"
  "/home/likelytobelaura/osteo/aat/build/kernel"
  "/home/likelytobelaura/osteo/aat/build/aat_kernel-prefix"
  "/home/likelytobelaura/osteo/aat/build/aat_kernel-prefix/tmp"
  "/home/likelytobelaura/osteo/aat/build/aat_kernel-prefix/src/aat_kernel-stamp"
  "/home/likelytobelaura/osteo/aat/build/aat_kernel-prefix/src"
  "/home/likelytobelaura/osteo/aat/build/aat_kernel-prefix/src/aat_kernel-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/likelytobelaura/osteo/aat/build/aat_kernel-prefix/src/aat_kernel-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/likelytobelaura/osteo/aat/build/aat_kernel-prefix/src/aat_kernel-stamp${cfgdir}") # cfgdir has leading slash
endif()

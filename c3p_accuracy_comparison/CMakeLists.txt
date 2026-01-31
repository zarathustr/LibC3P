cmake_minimum_required(VERSION 3.16)
project(pem_c3p_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Armadillo REQUIRED)

add_executable(pem_sim
    src/main.cpp
    src/se3.cpp
    src/lm.cpp
    src/problems.cpp
)
target_compile_options(pem_sim PRIVATE -O3 -w)
target_include_directories(pem_sim PRIVATE src /usr/local/include /usr/local/include/armadillo_bits)
target_link_libraries(pem_sim PRIVATE -L/usr/local/lib armadillo)

add_executable(pem_benchmark_mc
    src/benchmark_mc.cpp
    src/solvers.cpp
    src/se3.cpp
    src/lm.cpp
    src/problems.cpp
)
target_compile_options(pem_benchmark_mc PRIVATE -O3 -w)
target_include_directories(pem_benchmark_mc PRIVATE src /usr/local/include /usr/local/include/armadillo_bits)
target_link_libraries(pem_benchmark_mc PRIVATE -L/usr/local/lib armadillo)

add_executable(pem_c3p_noise_sweep
    src/c3p_noise_sweep.cpp
    src/solvers.cpp
    src/se3.cpp
    src/lm.cpp
    src/problems.cpp
)
target_compile_options(pem_c3p_noise_sweep PRIVATE -O3 -w)
target_include_directories(pem_c3p_noise_sweep PRIVATE src /usr/local/include /usr/local/include/armadillo_bits)
target_link_libraries(pem_c3p_noise_sweep PRIVATE -L/usr/local/lib armadillo)

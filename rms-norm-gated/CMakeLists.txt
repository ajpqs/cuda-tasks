add_catch(test_norm test.cpp norm.cu) 
add_catch(bench_norm run.cpp norm.cu)
target_link_libraries(test_norm PRIVATE ysda_cuda::util)
target_link_libraries(bench_norm PRIVATE ysda_cuda::util)
# silence junk warnings
target_compile_options(test_norm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --expt-relaxed-constexpr
                       >)
target_compile_options(bench_norm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --expt-relaxed-constexpr
                       >)
set_target_properties(test_norm PROPERTIES CUDA_ARCHITECTURES "75;86")
set_target_properties(bench_norm PROPERTIES CUDA_ARCHITECTURES "75;86")

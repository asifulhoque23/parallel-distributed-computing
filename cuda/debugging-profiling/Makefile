NVCC ?= nvcc

.PHONY: all clean
all: debug_bug debug_fixed math_bug math_fixed profile_demo transfer_demo

debug_bug: debug_demo.cu
	$(NVCC) -g -G -DDEMO_BUG $< -o $@

debug_fixed: debug_demo.cu
	$(NVCC) -g -G $< -o $@

profile_demo: profile_demo.cu
	$(NVCC) -O2 -lineinfo $< -o $@

math_bug: math_debug_demo.cu
	$(NVCC) -g -G -DDEMO_BAD_MATH $< -o $@

math_fixed: math_debug_demo.cu
	$(NVCC) -g -G $< -o $@

clean:
	rm -f debug_bug debug_fixed math_bug math_fixed profile_demo transfer_demo launch_bad launch_fixed

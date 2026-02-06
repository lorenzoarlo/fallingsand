# Default parameters values
FRAMES ?= 100
SCALE ?= 1
SAMPLE ?= 1
LOGIC ?= src/logic.c
CUDA_ARCHITECTURES ?= 75

# Default target
default:
	cmake -B build -DSIMULATION_LOGIC="${LOGIC}"
	cmake --build build
default-o3: 
	cmake -B build -DSIMULATION_LOGIC="${LOGIC}" -DCMAKE_BUILD_TYPE=Release
	cmake --build build
# Run simulation and generate video logic
video: default
	mkdir -p output
	rm -f output/images/sample-${SAMPLE}-*.png
	mkdir -p output/images/sample-${SAMPLE}
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -oi output/images/sample-${SAMPLE}/ -s ${SCALE} -l output/logs/sample-${SAMPLE}-performance.log
	mkdir -p output/videos
	rm -f output/videos/animation-${SAMPLE}.mp4
	ffmpeg -framerate 60 -i output/images/sample-${SAMPLE}/%04d.png -c:v libx264 -crf 0 -pix_fmt yuv420p output/videos/animation-${SAMPLE}.mp4
# Run simulation and generate video using default logic with 1500 frames and scale 4 (useful for high-res videos)
output_video: SCALE = 4
output_video: FRAMES = 1500
output_video: video

# Run simulation and compare output with reference solution (without image output)
test: default
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log

test-o3: default-o3
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log 


test-simd: LOGIC = src/simd/simd-optimized.cpp
test-simd: default
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log 


test-simd-o3: LOGIC = src/simd/simd-optimized.cpp
test-simd-o3: default-o3
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log 

test-simd-manual: LOGIC = src/simd/simd-optimized-manual-shuffle.cpp
test-simd-manual: default
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log 


test-simd-manual-o3: LOGIC = src/simd/simd-optimized.cpp
test-simd-manual-o3: default-o3
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log 

video-simd: LOGIC = src/simd/simd-optimized.cpp
video-simd: default
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -oi output/images/sample-${SAMPLE}/  -l output/logs/sample-${SAMPLE}-performance.log

performance-simd: LOGIC = src/simd/simd-optimized.cpp
performance-simd: default
	mkdir -p output
	mkdir -p output/logs/
	xcrun xctrace record --template "Time Profiler" --output simulazione.trace --launch -- ./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -s ${SCALE} -l output/logs/sample-${SAMPLE}-performance.log 

# Base CUDA target
default_cuda: 
	cmake -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DSIMULATION_LOGIC=${LOGIC} -DCUDA_ARCH=${CUDA_ARCHITECTURES}
	cmake --build build
default_test_cuda: default_cuda
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log

default_testdiff_cuda: default_cuda
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -s ${SCALE} -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log -oi output/images/sample-${SAMPLE}/

default_performance_ncu: default_cuda
	mkdir -p output
	mkdir -p output/ncu_reports/
	mkdir -p output/logs/
	ncu --set full --section SpeedOfLight_RooflineChart -o "output/ncu_reports/ncu_report_sample${SAMPLE}" ./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -s ${SCALE} -l output/logs/sample-${SAMPLE}-performance.log

default_performance_nvprof: default_cuda
	mkdir -p output
	mkdir -p output/ncu_reports/
	mkdir -p output/logs/
	rm -f output/images/sample-${SAMPLE}-*.png
	mkdir -p output/images/sample-${SAMPLE}
	nvprof --metrics achieved_occupancy,ipc,warp_execution_efficiency,gld_efficiency,gst_efficiency -o "output/ncu_reports/ncu_report_sample${SAMPLE}" ./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -s ${SCALE} -l output/logs/sample-${SAMPLE}-performance.log

video_cuda: default_cuda
	mkdir -p output
	rm -f output/images/sample-${SAMPLE}-*.png
	mkdir -p output/images/sample-${SAMPLE}
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -oi output/images/sample-${SAMPLE}/ -s ${SCALE} -l output/logs/sample-${SAMPLE}-performance.log
	mkdir -p output/videos
	rm -f output/videos/animation-${SAMPLE}.mp4
	ffmpeg -framerate 60 -i output/images/sample-${SAMPLE}/%04d.png -c:v libx264 -pix_fmt yuv420p output/videos/animation-${SAMPLE}.mp4

references: LOGIC = src/logic.c
references: default
	mkdir -p assets
	mkdir -p assets/references/
	mkdir -p assets/logs/
	./build/bin/fallingsand assets/sample-1.sand assets/references/output-sample-1.sand 1500 -l assets/logs/base-performance-1.csv
	./build/bin/fallingsand assets/sample-2.sand assets/references/output-sample-2.sand 1500 -l assets/logs/base-performance-2.csv

# Clean build artifacts 
clean:
	rm -rf build
	rm -rf bin
# Clean output files
clean_output:
	rm -rf output/*.sand
	rm -rf output/ncu_reports/*
	rm -rf output/images/**/*.png
	rm -rf output/images/**/*.ppm 
	rm -rf output/videos/*.mp4
	rm -rf output/logs/*.log
# Clean all
clean_all: clean clean_output
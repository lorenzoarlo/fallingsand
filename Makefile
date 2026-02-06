# Default parameters values
FRAMES ?= 1500
SCALE ?= 1
SAMPLE ?= 1
LOGIC ?= src/logic.c
CUDA_ARCHITECTURES ?= 75 # Needed by CUDA to compile for specific GPU architectures
FRAMERATE=120
# Default target
default:
	cmake -B build -DSIMULATION_LOGIC="${LOGIC}"
	cmake --build build
default-o3: 
	cmake -B build -DSIMULATION_LOGIC="${LOGIC}" -DCMAKE_BUILD_TYPE=Release
	cmake --build build

generate-references: LOGIC = src/logic.c
generate-references: default
	mkdir -p assets
	mkdir -p assets/references/
	mkdir -p assets/logs/
	./build/bin/fallingsand assets/sample-1.sand assets/references/output-sample-1.sand 1500 -l assets/logs/base-performance-1.csv
	./build/bin/fallingsand assets/sample-2.sand assets/references/output-sample-2.sand 1500 -l assets/logs/base-performance-2.csv
	./build/bin/fallingsand assets/sample-3.sand assets/references/output-sample-3.sand 3000 -l assets/logs/base-performance-3.csv

# Run simulation and generate video logic
video: default
	mkdir -p output
	rm -f output/images/sample-${SAMPLE}-*.png
	mkdir -p output/images/sample-${SAMPLE}
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -oi output/images/sample-${SAMPLE}/ -s ${SCALE} -l output/logs/sample-${SAMPLE}-performance.csv
	mkdir -p output/videos
	rm -f output/videos/animation-${SAMPLE}.mp4
	ffmpeg -framerate ${FRAMERATE} -i output/images/sample-${SAMPLE}/%04d.png -c:v libx264 -crf 0 -pix_fmt yuv420p output/videos/animation-${SAMPLE}.mp4


capture-original: LOGIC = src/utility/freeze.c
capture-original: default
	mkdir -p output
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 10 -oi output/images/sample-${SAMPLE}/ -s 4 -l /dev/null

# Run simulation and compare output with reference solution (without image output)
test: default
	mkdir -p output
	mkdir -p statistics
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -t assets/references/output-sample-${SAMPLE}.sand -l statistics/sequential-sample-${SAMPLE}.csv

test-o3: default-o3
	mkdir -p output
	mkdir -p statistics
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -t assets/references/output-sample-${SAMPLE}.sand -l statistics/sequential-o3-sample-${SAMPLE}.csv


test-simd-manual-interleave: LOGIC = src/simd/simd-manual-interleave.cpp
test-simd-manual-interleave: test

test-simd-manual-interleave-prefetch: LOGIC = src/simd/simd-manual-interleave-prefetch.cpp
test-simd-manual-interleave-prefetch: test

test-simd-manual-interleave-prefetch-o3: LOGIC = src/simd/simd-manual-interleave-prefetch.cpp
test-simd-manual-interleave-prefetch-o3: test-o3

# Profiling for SIMD version
performance-simd: LOGIC = src/simd/simd-manual-interleave.cpp
performance-simd: default
	mkdir -p output
	mkdir -p output/logs/
	xcrun xctrace record --template "Time Profiler" --output simulazione.trace --launch -- ./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -s ${SCALE} -l output/logs/sample-${SAMPLE}-performance.csv 

# Base CUDA target
default-cuda: 
	cmake -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc -DSIMULATION_LOGIC=${LOGIC} -DCUDA_ARCH=${CUDA_ARCHITECTURES}
	cmake --build build
test-cuda: default-cuda
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/cuda-sample-${SAMPLE}-performance.csv

cuda-profiling: default-cuda
	mkdir -p output
	mkdir -p output/ncu-reports/
	mkdir -p output/logs/
	ncu --set full --section SpeedOfLight_RooflineChart -o "output/ncu-reports/ncu-report-sample-${SAMPLE}" ./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -s ${SCALE} -l /dev/null

video-cuda: default-cuda
	mkdir -p output
	rm -f output/images/sample-${SAMPLE}-*.png
	mkdir -p output/images/sample-${SAMPLE}
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -oi output/images/sample-${SAMPLE}/ -s ${SCALE} -l /dev/null
	mkdir -p output/videos
	rm -f output/videos/animation-${SAMPLE}.mp4
	ffmpeg -framerate ${FRAMERATE} -i output/images/sample-${SAMPLE}/%04d.png -c:v libx264 -crf 0 -pix_fmt yuv420p output/videos/animation-${SAMPLE}.mp4


# Clean build artifacts 
clean:
	rm -rf build
	rm -rf bin
# Clean output files
clean-output:
	rm -rf output/*.sand
	rm -rf output/ncu-reports/*
	rm -rf output/images/**/*.png
	rm -rf output/images/**/*.ppm 
	rm -rf output/videos/*.mp4
	rm -rf output/logs/*.csv
	rm -rf statistics/*
# Clean all
clean-all: clean clean-output
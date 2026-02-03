# Default number of frames to simulate
FRAMES ?= 100
SCALE ?= 1
SAMPLE ?= 1
LOGIC ?= src/logic.c
default:
	cmake -B build -DSIMULATION_LOGIC="${LOGIC}"
	cmake --build build
video: default
	mkdir -p output
	rm -f output/images/sample-${SAMPLE}-*.png
	mkdir -p output/images/sample-${SAMPLE}
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -oi output/images/sample-${SAMPLE}/ -s ${SCALE} -l output/logs/sample-${SAMPLE}-performance.log
	mkdir -p output/videos
	rm -f output/videos/animation-${SAMPLE}.mp4
	ffmpeg -framerate 90 -i output/images/sample-${SAMPLE}/%04d.png -c:v libx264 -crf 0 -pix_fmt yuv420p output/videos/animation-${SAMPLE}.mp4
# Run simulation and generate video using default logic with 1500 frames and scale 4 (useful for high-res videos)
default_video: SCALE = 4
default_video: FRAMES = 1500
default_video: video

default_test: default
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log

default_cuda: 
	cmake -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc
	cmake --build build
default_test_cuda: default_cuda
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log

default_testdiff_cuda: default_cuda
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -s ${SCALE} -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log -oi output/images/sample-${SAMPLE}/

default_performace_ncu: default_cuda
	mkdir -p output
	mkdir -p output/ncu_reports/
	ncu --set full -o "output/ncu_reports/ncu_report_sample${SAMPLE}" ./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -oi output/images/sample-${SAMPLE}/ -s ${SCALE} -l output/logs/sample-${SAMPLE}-performance.log

video_cuda: default_cuda
	mkdir -p output
	rm -f output/images/sample-${SAMPLE}-*.png
	mkdir -p output/images/sample-${SAMPLE}
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -oi output/images/sample-${SAMPLE}/ -s ${SCALE} -l output/logs/sample-${SAMPLE}-performance.log
	mkdir -p output/videos
	rm -f output/videos/animation-${SAMPLE}.mp4
	ffmpeg -framerate 60 -i output/images/sample-${SAMPLE}/%04d.png -c:v libx264 -pix_fmt yuv420p output/videos/animation-${SAMPLE}.mp4

less_effort_test: LOGIC = src/simd/less-effort-logic.c
less_effort_test: default
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log

clean:
	rm -rf build
	rm -rf bin
clean_output:
	rm -rf output/*.sand
	rm -rf output/images/**/*.png
	rm -rf output/images/**/*.ppm 
	rm -rf output/videos/*.mp4
	rm -rf output/logs/*.log

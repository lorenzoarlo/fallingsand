# Default number of frames to simulate
FRAMES ?= 100
SCALE ?= 1
SAMPLE ?= 1
LOGIC ?= src/logic.c
default:
	cmake -B build -DSIMULATION_LOGIC="${LOGIC}"
	cmake --build build
# Run simulation and generate video using default logic with 1500 frames and scale 4 (useful for high-res videos)
default_video: default
	mkdir -p output
	rm -f output/images/sample-${SAMPLE}-*.png
	mkdir -p output/images/sample-${SAMPLE}
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -oi output/images/sample-${SAMPLE}/ -s 4 -l output/logs/sample-${SAMPLE}-performance.log
	mkdir -p output/videos
	rm -f output/videos/animation-${SAMPLE}.mp4
	ffmpeg -framerate 60 -i output/images/sample-${SAMPLE}/%04d.png -c:v libx264 -pix_fmt yuv420p output/videos/animation-${SAMPLE}.mp4
default_test: default
	mkdir -p output
	mkdir -p output/logs/
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand 1500 -t assets/references/output-sample-${SAMPLE}.sand -l output/logs/sample-${SAMPLE}-performance.log
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

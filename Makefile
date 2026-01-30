# Default number of frames to simulate
FRAMES ?= 100
SCALE ?= 3
SAMPLE ?= 1

default:
	cmake -S . -B build
	cmake --build build
default_video: default
	mkdir -p output
	rm -f output/images/sample-${SAMPLE}-*.png
	mkdir -p output/images/sample-${SAMPLE}
	./build/bin/fallingsand assets/sample-${SAMPLE}.sand output/output-sample-${SAMPLE}.sand ${FRAMES} -oi output/images/sample-${SAMPLE}/ -s ${SCALE}
	mkdir -p output/videos
	rm -f output/videos/animation-${SAMPLE}.mp4
	ffmpeg -framerate 60 -i output/images/sample-${SAMPLE}/%04d.png -c:v libx264 -pix_fmt yuv420p output/videos/animation-${SAMPLE}.mp4
clean:
	rm -rf build
	rm -rf bin
clean_output:
	rm -rf output/*.sand
	rm -rf output/images/**/*.png
	rm -rf output/images/**/*.ppm 
	rm -rf output/videos/*.mp4

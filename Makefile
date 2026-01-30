default:
	cmake -S . -B build
	cmake --build build
default_video: default
	mkdir -p output
	rm -f output/images/*.png
	./build/bin/fallingsand assets/sample-1.sand output/output-sample-1.sand 100 -oi output/images/ -s 3
	rm -f output/videos/animation.mp4
	ffmpeg -framerate 60 -i output/images/%04d.png -c:v libx264 -pix_fmt yuv420p output/videos/animation.mp4
clean:
	rm -rf build
	rm -rf bin
clean_output:
	rm -rf output/*.sand
	rm -rf output/images/*.png
	rm -rf output/images/*.ppm 
	rm -rf output/videos/animation.mp4

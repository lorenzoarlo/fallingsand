default:
	cmake -S . -B build
	cmake --build build
default_video:
	make default
	./bin/fallingsand assets/sample-1.sand output/output-sample-1.sand 100 -oi output/ -s 3
	ffmpeg -framerate 60 -i output/%04d.ppm -c:v libx264 -pix_fmt yuv420p output/animation.mp4

clean:
	rm -rf build
	rm -rf bin
# Falling sand project

The falling sand project is a simulation of particles that interact with each other based on simple physical rules. The main types of particles are empty space, walls, sand, and water. Each particle type has its own behavior, such as falling under gravity or flowing.

## Algorithm overview

The simulation consists of four distinct particle types, each represented by a specific integer value (indicating relative density):

| Particle Name | Value | Description                           |
| ------------- | ----- | ------------------------------------- |
| `EMPTY`     | `0`   | Represents empty space.               |
| `WATER`     | `1`   | Affects gravity; flows horizontally.  |
| `SAND`      | `2`   | Affects gravity; sinks through water. |
| `WALL`      | `3`   | An immovable block.                   |


### Base structure

The base structure for the simulation universe is defined as follows:

```c
typedef struct {
    unsigned char *cells; // Pointer to a dynamic array of size width * height
    int width;
    int height;
} Universe;
```

The `*cells` array stores the grid rows sequentially (row-major order), starting from the top row.

Any coordinate outside the valid matrix dimensions is considered a `WALL`.

### Function Prototype

The performance and logic of the following function will be analyzed:

```c
next(Universe* universe, Universe* out, int generation) {
    // Implementation logic here
}
```

The goal is returns a pointer to a data structure containing the `Universe` state for the next generation.

To ensure consistency, the input data structure can be modified directly (in-place) without allocating a new instance, though allocating a new instance is also permitted.

The **generation** represents the current frame number being generated.

### `next` function logic

The `next` function calculates the configuration for the subsequent generation by iterating through all cells.

## Change the simulation logic

To make it interchangeable is used a cmake file that can substitute the logic implementation with another one. By default the "naive" logic is used (it uses the `logic.c`).

```bash
cmake . && make
```

To use a different logic implementation, specify it when running cmake. For example, to use the "optimized" logic:

```bash
cmake -DSIMULATION_LOGIC="src/file.c" . && make
```

To run the simulation (with a test reference):

```bash
./bin/fallingsand input.sand output.sand 30 -oi output_folder -t test_reference.sand
```

### Default logic implementation

For example

```bash
./bin/fallingsand assets/sample-1.sand output/output-sample-1.sand 500 -oi output/
```
or you can build it with

```bash
make default_video
```

### Freeze logic implementation

```bash
cmake -DSIMULATION_LOGIC="src/utility/freeze.c" . # ...
```

## Make file commands
To help you with building and running the simulation, the following make commands are available:

```bash
make default # Build with default logic implementation
```
Other useful commands are:

```bash
make default_video # Build and run with default logic implementation and generate a video
```
Eventually, you can also add some parameters to the command:
```bash
make default_video SAMPLE=sample FRAMES=number_of_frames SCALE=factor
```
where:
- `sample` is which original state to use (default: `1`, corresponds to `assets/sample-1.sand`);
- `number_of_frames` is how many frames to simulate (default: `100`);
- `factor` is the scaling factor for the output images (default: `1`).  


```bash
make default_test # Build and run with default logic implementation and test it against reference
```

About references, by default it expected files to be in `assets/references/` folder and logs will be saved in `output/logs/` folder.
About names, they are by default like `output-sample-{SAMPLE}.sand`.


## Create a video 
It is possible to create a video with all the frames using ffmpeg
```bash
ffmpeg -framerate 60 -i output/iages/sample-1/%04d.png -c:v libx264 -pix_fmt yuv420p output/animation.mp4
```

- [Falling sand project](#falling-sand-project)
  - [Algorithm overview](#algorithm-overview)
    - [Base structure](#base-structure)
    - [Function Prototype](#function-prototype)
    - [`next` function logic](#next-function-logic)
      - [Iteration Order](#iteration-order)
      - [Particle behaviors](#particle-behaviors)
        - [`EMPTY` and `WALL`](#empty-and-wall)
        - [`SAND`](#sand)
        - [`WATER`](#water)
  - [Change the simulation logic](#change-the-simulation-logic)
  - [Make file commands](#make-file-commands)
  - [Create a video](#create-a-video)


# Falling sand project

The falling sand project is a simulation of particles that interact with each other based on simple physical rules. The main types of particles are empty space, walls, sand, and water. Each particle type has its own behavior, such as falling under gravity or flowing.

## Algorithm overview

The simulation consists of four distinct particle types, each represented by a specific integer value:

| Particle Name | Value | Description                           |
| ------------- | ----- | ------------------------------------- |
| `EMPTY`     | `0`   | Represents empty space.               |
| `WALL`      | `1`   | An immovable block.                   |
| `SAND`      | `2`   | Affects gravity; sinks through water. |
| `WATER`     | `3`   | Affects gravity; flows horizontally.  |

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

#### Iteration Order

The grid must be traversed from **Top to Bottom**. The horizontal direction of traversal depends on the generation number:

- **Even Generation** (`generation % 2 == 0`): Iterate from **Left Right**.
- **Odd Generation** (`generation % 2 != 0`): Iterate from **Right Left**.

If a particle is already been updated in the current generation, it **should not be processed again**.

#### Particle behaviors

##### `EMPTY` and `WALL`

- These particles remain static; no action is taken.

##### `SAND`

Behavior depends on the environment immediately below the particle.
It is needed to check if the "targeted cell" is already updated. 
If so, in a double buffer implementation, it is necessary to check the "targeted cell" of the "next universe".

> CHANGES: Specified this behavior more clearly.

1. **If the cell below is `EMPTY`:**
- Move the `SAND` particle down.


2. **If the cell below is blocked (not `EMPTY`):**
- Attempt to slide diagonally into an `EMPTY` space.
- The check order depends on **generation parity** (to create random distribution):
- *Even Generation:* Check **Left** first, then **Right**.
- *Odd Generation:* Check **Right** first, then **Left**.

- If the primary diagonal is `EMPTY`, move there.
- If the primary is blocked, check the secondary diagonal.


3. **If the cell below is `WATER` (and movement was not possible above):**
- Apply a **viscosity check** (`(x + y + generation) % 2 != 0`, almost 50%).
- If the check passes (~50% chance), **swap positions** with the `WATER` particle (simulating density).
- Otherwise, stay in place 

> this prevents `WATER` from displacing upwards too rapidly

4. **Else:**
- Stay in the current position.

> CHANGES: Changed logic of `SAND`.

##### `WATER`

Behavior depends on the environment immediately below the particle.
It is needed to check if the "targeted cell" is already updated. 
If so, in a double buffer implementation, it is necessary to check the "targeted cell" of the "next universe".

1. **Vertical Movement:**

- If the cell below is `EMPTY`, move down.

2. **Diagonal Movement:**

- If the cell below is `WALL` or `SAND`, check the diagonals (looking for `EMPTY` spots) in the following order:
- _Even Generation:_ Check **Bottom-Left**, then **Bottom-Right**.
- _Odd Generation:_ Check **Bottom-Right**, then **Bottom-Left**.

3. **Horizontal Movement (Flow):**

- If the diagonals are also occupied, check horizontal neighbors:
- _Even Generation:_ Check **Right**, then **Left**.
- _Odd Generation:_ Check **Left**, then **Right**.

4. **Stationary:**

- If all checked cells are occupied, remain stationary.

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
To run it with the default logic implementation and generate a video:

```bash
make default_video
```
Eventually, you can also add some parameters to the command:

```bash
make default_video SAMPLE=sample FRAMES=number_of_frames SCALE=factor
```
where:
- `sample` is which original state to use (default: `1`, corresponds to `assets/sample-1.sand`);
- `number_of_frames` is how many frames to simulate (default: `100`);
- `factor` is the scaling factor for the output images (default: `1`).  

## Makefile commands
To help you with building and running the simulation, the following make commands are available:

```bash
make default_video # Build and run with default logic implementation and generate a video
```

```bash
make default_test # Build and run with default logic implementation and test it against reference
```

```bash
make less_effort_test # Build and run with less-effort logic implementation and test it against reference
```

About references, by default it expected files to be in `assets/references/` folder and logs will be saved in `output/logs/` folder.
About names, they are by default like `output-sample-{SAMPLE}.sand`.


## Create a video 
It is possible to create a video with all the frames using ffmpeg
```bash
ffmpeg -framerate 60 -i output/%04d.ppm -c:v libx264 -pix_fmt yuv420p output/animation.mp4
```
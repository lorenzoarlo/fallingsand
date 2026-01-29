- [Falling sand project](#falling-sand-project)
    - [Base structure](#base-structure)
    - [Function Prototype](#function-prototype)
    - [`next` function logic](#next-function-logic)
      - [Iteration Order](#iteration-order)
      - [Particle behaviors](#particle-behaviors)
        - [`EMPTY` and `WALL`](#empty-and-wall)
        - [`SAND`](#sand)
        - [`WATER`](#water)
  - [Change the simulation logic](#change-the-simulation-logic)


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

`c
typedef struct {
    unsigned char *cells; // Pointer to a dynamic array of size width * height
    int width;
    int height;
} Universe;

`

The `*cells` array stores the grid rows sequentially (row-major order), starting from the top row.

Any coordinate outside the valid matrix dimensions is considered a `WALL`.

### Function Prototype

The performance and logic of the following function will be analyzed:

```c
Universe* next(Universe* universe, int generation) {
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

#### Particle behaviors

##### `EMPTY` and `WALL`

- These particles remain static; no action is taken.

##### `SAND`

Behavior depends on the environment immediately below the particle:

1. **If the cell below is `EMPTY`:**

- Move the `SAND` particle down.

2. **If the cell below is `WATER`:**

- **Swap positions** with the `WATER` particle (simulating density).

3. **If the cell below is `SAND` or `WALL`:**

- Check **one** diagonal based on the generation parity:
- _Even Generation:_ Check the **Bottom-Left** diagonal.
- _Odd Generation:_ Check the **Bottom-Right** diagonal.

- If the checked diagonal is `EMPTY`, move there else remain stationary. (Note: Do _not_ check the other diagonal, and do not swap if the diagonal contains `WATER`).

##### `WATER`

Behavior depends on the environment immediately below the particle:

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

`bash
cmake . && make
`

To use a different logic implementation, specify it when running cmake. For example, to use the "optimized" logic:

```bash
cmake -DSIMULATION_LOGIC="src/optimized.c" . && make
```

To run the simulation (with a test reference):

```bash
./bin/fallingsand input.sand output.sand 30 -oi output_folder -t test_reference.sand
`

For example

```bash
./bin/fallingsand assets/sample-1.sand output/output-sample-1.sand 30 -oi output/
```

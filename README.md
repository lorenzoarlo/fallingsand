# Falling sand project

The falling sand project is a simulation of particles that interact with each other based on simple physical rules. The main types of particles are empty space, walls, sand, and water. Each particle type has its own behavior, such as falling under gravity or flowing.

## Logic
To make it interchangeable is used a cmake file that can substitute the logic implementation with another one. By default the "naive" logic is used (it uses the `logic.c`).

```bash
cmake . && make
```

To use a different logic implementation, specify it when running cmake. For example, to use the "optimized" logic:

```bash
cmake -DSIMULATION_LOGIC="src/optimized.c" . && make
```

To run the simulation (with a test reference):

```bash
./bin/fallingsand input.sand output.sand 30 -oi output_folder -t test_reference.sand
```

For example

```bash
./bin/fallingsand assets/sample-1.sand output/output-sample-1.sand 30 -oi output/
```

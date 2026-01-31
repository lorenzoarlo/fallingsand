/**
 * @file performance.c
 * Implementation of performance.h useful for both architectures
 */
#include <stdint.h>
#include <stdio.h>

void append_performance(FILE *output, uint64_t start, uint64_t end, int frame)
{
    // Append frame number and cycles taken
    // frame;start;end;cycles
    fprintf(output, "%d;%llu;%llu;%llu\n", frame, start, end, (end - start));
}
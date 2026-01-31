/**
 * @file performance.h
 * @brief Definitions of performance files
 */
#include <stdint.h>
#include <stdio.h>

#if defined(__i386__) || defined(__x86_64__)
#define ARCH_X86
#elif defined(__aarch64__)
#define ARCH_ARM64
#else
#error "Architecture not supported"
#endif

/*
 * Implementation x86
 */
#ifdef ARCH_X86
/**
 * @brief Measure start time on x86
 */
static inline uint64_t measure_start(void)
{
    uint32_t cycles_high, cycles_low;
    // volatile tells the compilers to manage correctly
    asm volatile(
        "cpuid\n\t"
        "rdtsc\n\t"
        "mov %%edx, %0\n\t"
        "mov %%eax, %1\n\t"
        : "=r"(cycles_high), "=r"(cycles_low)
        :
        : "%rax", "%rbx", "%rcx", "%rdx");

    return ((uint64_t)cycles_high << 32) | cycles_low;
}

/**
 * @brief Measure end time on x86
 */
static inline uint64_t measure_end(void)
{
    uint32_t cycles_high, cycles_low;

    asm volatile(
        "rdtscp\n\t"
        "mov %%edx, %0\n\t"
        "mov %%eax, %1\n\t"
        "cpuid\n\t"
        : "=r"(cycles_high), "=r"(cycles_low)
        :
        : "%rax", "%rbx", "%rcx", "%rdx");

    return ((uint64_t)cycles_high << 32) | cycles_low;
}

#endif

/*
 * ARM implementation for macos
 */
#ifdef ARCH_ARM64

/**
 * @brief Measure start time on ARM64
 */
static inline uint64_t measure_start(void)
{
    uint64_t val;
    asm volatile(
        "isb\n\t"
        "mrs %0, cntvct_el0"
        : "=r"(val)
        :
        : "memory");
    return val;
}

/**
 * @brief Measure end time on ARM64
 */
static inline uint64_t measure_end(void)
{
    uint64_t val;
    asm volatile(
        "isb\n\t"
        "mrs %0, cntvct_el0\n\t"
        "isb\n\t"
        : "=r"(val)
        :
        : "memory");
    return val;
}

#endif
/**
 * @brief Append performance data to output file
 */
void append_performance(FILE *output, uint64_t start, uint64_t end, int frame);
/*
 * main.cu
 *
 * This file contains the code for our final project, designed to simulate light propagation through
 * a medium using Monte Carlo methods.
 *
 * The original code was written by Scott Prahl and can be found here:
 * https://omlc.org/software/mc/small_mc.c
 *
 * Author: Scott Prahl, Jonathan Paulick, Jonathan Phung
 * Creation Date: 10/22/22
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <curand_kernel.h>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// Absorption Coefficient in 1/cm
#define MU_A 5.0

// Scattering Coefficient in 1/cm
#define MU_S 95.0

// Scattering Anisotropy -1 <= g <= 1
#define G 0.5

// Index of refraction of medium
#define N 1.5

// Thickness of one bin layer
#define MICRONS_PER_BIN 20.0

// Number of NUMBER_OF_PHOTONS to simulate
#define NUMBER_OF_PHOTONS 100000l

// Specular reflection
#define RS ((N - 1.0) * (N - 1.0) / (N + 1.0) / (N + 1.0))

#define ALBEDO (MU_S / (MU_S + MU_A))

// Cosine of critical angle
#define CRITICAL_ANGLE (sqrt(1.0 - 1.0 / N / N))

#define BINS_PER_MFP (1e4 / MICRONS_PER_BIN / (MU_A + MU_S))

#define BINS 101

#define SEED 1

// The total number of photons that can be generated on one gpu
#define MAXPHOTONS 6144

// The maximum number of threads that can be used in a kernel
#define THREADS 1024

/**
 * @struct Photon
 * @brief This struct represents a photon to be simulated.
 * @var Photon::xPosition The x-position of the photon, formerly x.
 * @var Photon::yPosition The y-position of the photon, formerly y.
 * @var Photon::zPosition The z-position of the photon, formerly z.
 * @var Photon::deltaXPosition The change in the x-position of the photon, formerly u.
 * @var Photon::deltaYPosition The change in the y-position of the photon, formerly v.
 * @var Photon::deltaZPosition The change in the z-position of the photon, formerly w.
 * @var Photon::weight The weight of the photon.
 */

typedef struct {
    double xPosition;
    double yPosition;
    double zPosition;
    double deltaXPosition;
    double deltaYPosition;
    double deltaZPosition;
    double weight;
} Photon;

/**
 * @brief This function measures the elapsed, wall-clock time on the host given two timespecs in
 * milliseconds and returns it.  This was given as a part of Lab 1 and has been cleaned for
 * consistency.
 * @param start A pointer to the timespec representing the start of timing.
 * @param end A pointer to the timespec representing the end of timing.
 * @return The elapsed, wall-clock time between the given end and start timespecs in milliseconds.
 */

float host_time(struct timespec* start, struct timespec* end);

/**
 * @brief Code given to us to handle CUDA errors 
 * @param err A cuda error to handle
 * @param file A pointer to the current file
 * @param line The line which the error occurred
 */

static void HandleError(cudaError_t err, const char *file, int line);

/**
 * @brief This function initializes the given Photon to the starting values.
 * @param photon A pointer to the Photon to initialize.
 */

__device__ void initialize_photon(Photon *photon);

/**
 * @brief This function simulates the bouncing of a photon off the top surface.
 * @param photon A pointer to the Photon to simulate.
 * @param rd A statistic used in simulation
 */

__device__ void bounce(Photon* photon, double* rd);

/**
 * @brief This function simulates the movement to next scattering or absorption event
 * @param photon A pointer to the Photon to simulate.
 * @param rd A statistic used in simulation
 * @param state The status of the cuRand function for random number generation  
 */
__device__ void move(Photon* photon, double* rd, curandState* state);

/**
 * @brief This function simulates the absorption of light into the medium
 * @param photon A pointer to the Photon to simulate.
 * @param bit A statistic used to calculate the results
 * @param heat An array of statistics used to calculate the results
 * @param state The status of the cuRand function for random number generation  
 */
__device__ void absorb(Photon* photon, double* bit, double heat[], curandState* state);

/**
 * @brief This function simulates the scattering of a photon and establish a new direction
 * @param photon A pointer to the Photon to initialize.
 * @param state The status of the cuRand function for random number generation  
 */
__device__ void scatter(Photon* photon,  curandState* state);

/**
 * @brief This function will print the results and statistics of the Monte Carlo simulations.
 * @param rd A statistic used to calculate the results
 * @param bit A statistic used to calculate the results
 * @param heat An array of statistics used to calculate the results
 * @param totalPhotons The total number of generated photons in the simulation
 */

void print_results(double* rd, double* bit, double heat[], long totalPhotons);

/**
 * @brief A helper method to generate random numbers between 0 and MAX_RAND
 * @param state The status of the cuRand function for random number generation  
 * @return A randomly generated number between 0 and MAX_RAND
 */

__device__ int random(curandState* state);

/**
 * @brief Kernel for generating movement with photons 
 * @param d_rd A pointer to an array in global memory. Is a statistic used to calculate the results
 * @param d_bit A pointer to an array in global memory. Is a statistic used to calculate the results
 * @param d_heat A pointer to a 2d array in global memory. Is used to calculate the results
 * @param photons The number of photons being simulated 
 */

__global__ void simulationKernel(double* d_rd, double* d_bit, double* d_heat, int photons);

/**
 * @brief Kernel for reducing all elements in data into grid number of elements
 *   This version uses n/2 threads --
 *   it performs the first level of reduction when reading from global memory.
 *   Heavily inspired by reduce3 - https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu
 * @param data A pointer to an array in global memory 
 * @param n The number of elements in the data array
 */

__global__ void reduceKernel(double* data, int n);

/**
 * @brief Calls the reduceKernel for d_rd, d_bit, and d_heat until one value is stored in the first 
 * memory location
 * @param d_rd A pointer to an array in global memory. Is a statistic used to calculate the results
 * @param d_bit A pointer to an array in global memory. Is a statistic used to calculate the results
 * @param d_heat A pointer to a 2d array in global memory. Is used to calculate the results
 * @param offset The starting location in memory for the reduction to take place
 * @param photons The number of photons being simulated 
 * @param totalPhotons The number of photons stored in each array
 */

void reducer(double* d_rd, 
             double* d_bit, 
             double d_heat[], 
             long offset, 
             long photons, 
             long totalPhotons);

/**
 * @brief Method for starting the simulation and combining the outputs
 * @param h_rd A pointer to an array in host memory. Is a statistic used to calculate the results
 * @param h_bit A pointer to an array in host memory. Is a statistic used to calculate the results
 * @param h_heat A pointer to a 2d array in host memory. Is used to calculate the results
 * @param totalPhotons The number of photons being simulated 
 */

void gpu_simulation(double* h_rd, double* h_bit, double h_heat[], long totalPhotons);


int main(int argc, char* argv[]) {
    double rd = 0.0;
    double bit = 0.0;
    double heat[BINS] = {0};
    long totalPhotons;

    struct timespec ts;
    struct timespec te;

    errno = 0;
    char *p;

    if (argc > 1) {
        totalPhotons = strtol(argv[1], &p, 10);

        if (errno != 0 || *p != '\0') {
            fprintf( stderr, "Invalid number of photons");

            return 1;
        }
    } else {
        totalPhotons = NUMBER_OF_PHOTONS;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

    int photons = totalPhotons;

    while (photons > 10000000) {
        photons -= 10000000;
        gpu_simulation(&rd, &bit, heat, 10000000);
    }

    gpu_simulation(&rd, &bit, heat, photons);
    clock_gettime(CLOCK_MONOTONIC_RAW, &te);

    printf("Host Photon Simulation Runtime (ms): %f\n\n", host_time(&ts, &te));

    print_results(&rd, &bit, heat, totalPhotons);

    return 0;
}

/**
 * @brief A helper method to generate random numbers between 0 and MAX_RAND
 * @param state The status of the cuRand function for random number generation  
 */

__device__ int random(curandState* state) {
    return (int) (curand_uniform(state) * (RAND_MAX));
}

/**
 * @brief This function measures the elapsed, wall-clock time on the host given two timespecs in
 * milliseconds and returns it.  This was given as a part of the lab and has been cleaned for
 * consistency.
 * @param start A pointer to the timespec representing the start of timing.
 * @param end A pointer to the timespec representing the end of timing.
 * @return The elapsed, wall-clock time between the given end and start timespecs in milliseconds.
 */
float host_time(struct timespec* start, struct timespec* end) {
    return ((1e9 * end->tv_sec + end->tv_nsec) - (1e9 * start->tv_sec + start->tv_nsec)) / 1e6;
}

/**
 * @brief Code given to us to handle CUDA errors 
 * @param err A cuda error to handle
 * @param file A pointer to the current file
 * @param line The line which the error occurred
 */

static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
  }
}

/**
 * @brief This function initializes the given Photon to the starting values.
 * @param photon A pointer to the Photon to initialize.
 */

__device__ void initialize_photon(Photon *photon) {
    photon->xPosition = 0;
    photon->yPosition = 0;
    photon->zPosition = 0;
    photon->deltaXPosition = 0;
    photon->deltaYPosition = 0;
    photon->deltaZPosition = 1.0;
    photon->weight = 1 - RS;
}

/**
 * @brief This function simulates the bouncing of a photon off the top surface.
 * @param photon A pointer to the Photon to simulate.
 * @param rd A statistic used in simulation
 */

__device__ void bounce(Photon* photon, double* rd) {
    double t;
    double temp;
    double temp1;
    double rf;

    photon->deltaZPosition = -photon->deltaZPosition;
    photon->zPosition = -photon->zPosition;

    // This conditional checks the total internal reflection of the photon
    if(photon->deltaZPosition > CRITICAL_ANGLE) {
        // Cosine of exit angle
	t = sqrt(1.0 - N * N * (1.0 - photon->deltaZPosition * photon->deltaZPosition));

      	temp = (t - N * photon->deltaZPosition) / (t + N * photon->deltaZPosition);
        temp1 = (photon->deltaZPosition - N * t) / ((photon->deltaZPosition + N * t));

	// Fresnel reflection
        rf = (temp1 * temp1 + temp * temp) / 2.0;
	(*rd) += (1.0 - rf) * photon->weight;
	photon->weight -= (1.0 - rf) * photon->weight;
    }
}

/**
 * @brief This function simulates the movement to next scattering or absorption event
 * @param photon A pointer to the Photon to simulate.
 * @param rd A statistic used in simulation
 * @param state The status of the cuRand function for random number generation  
 */

__device__ void move(Photon* photon, double* rd, curandState* state) {
    double d = -log((random(state) + 1.0) / (RAND_MAX + 1.0));

    photon->xPosition += d * photon->deltaXPosition;
    photon->yPosition += d * photon->deltaYPosition;
    photon->zPosition += d * photon->deltaZPosition;

    /*
     * This photon checks if the photon's z position is less than or equal to 0 and bounces the
     * photon if it is.
     */

    if(photon->zPosition <= 0) {
        bounce(photon, rd);
    }
}

/**
 * @brief This function simulates the absorption of light into the medium
 * @param photon A pointer to the Photon to simulate.
 * @param bit A statistic used to calculate the results
 * @param heat An array of statistics used to calculate the results
 * @param state The status of the cuRand function for random number generation  
 */

__device__ void absorb (Photon* photon, double* bit, double heat[], curandState* state) {
    int bin = photon->zPosition * BINS_PER_MFP;

    if(bin >= BINS) {
        bin = BINS-1;
    }

    heat[bin] += (1.0 - ALBEDO) * photon->weight;
    photon->weight *= ALBEDO;

    // These nested if statement handles the roulette for photons with weights less than 0.001
    if(photon->weight < 0.001) {
        (*bit) -= photon->weight;

        if(random(state) > 0.1 * RAND_MAX) {
            photon->weight = 0;
        } else {
            photon->weight /= 0.1;
        }

        (*bit) += photon->weight;
    }
}

/**
 * @brief This function simulates the scattering of a photon and establish a new direction
 * @param photon A pointer to the Photon to initialize.
 * @param state The status of the cuRand function for random number generation  
 */

__device__ void scatter(Photon* photon, curandState* state) {
    double x1;
    double x2;
    double x3;
    double t;
    double mu;

    x1 = 2.0 * random(state) / RAND_MAX - 1.0;
    x2 = 2.0 * random(state) / RAND_MAX - 1.0;
    x3 = x3 = x1 * x1 + x2 * x2;

    // This while loop randomly chooses a new direction for the photon
    while(x3 <= 1) {
        x1 = 2.0 * random(state) / RAND_MAX - 1.0;
        x2 = 2.0 * random(state) / RAND_MAX - 1.0;
        x3 = x3 = x1 * x1 + x2 * x2;
    }

    // This if statement handles the case where g == 0, also known as the isotropic case
    if(G == 0) {
        photon->deltaXPosition = 2.0 * x3 - 1.0;
	    photon->deltaYPosition =
            x1 * sqrt((1 - photon->deltaXPosition * photon->deltaXPosition) / x3);
	    photon->deltaZPosition =
        x2 * sqrt((1 - photon->deltaXPosition * photon->deltaXPosition)  /x3);
    } else {
        mu = (1 - G * G) / (1 - G + 2.0 * G * random(state) / RAND_MAX);
        mu = (1 + G * G - mu * mu) / 2.0 / G;

        // This if statement checks if the speed of the photon in the z-direction is less than 0.9
        if(fabs(photon->deltaZPosition) < 0.9) {
            t = mu *
                photon->deltaXPosition +
                sqrt((1 - mu * mu) /
                (1 - photon->deltaZPosition * photon->deltaZPosition) / x3) *
                (x1 * photon->deltaXPosition * photon->deltaZPosition - x2 * photon->deltaYPosition);

      	    photon->deltaYPosition = mu *
                                     photon->deltaYPosition +
                                     sqrt((1 - mu * mu) / (1 - photon->deltaZPosition * photon->deltaZPosition) / x3) *
                                     (x1 * photon->deltaYPosition * photon->deltaZPosition + x2 *photon->deltaXPosition);

            photon->deltaZPosition = mu *
                                     photon->deltaZPosition -
                                     sqrt((1 - mu * mu) *
                                     (1 - photon->deltaZPosition * photon->deltaZPosition) / x3) *
                                     x1;

        } else {
            t = mu *
                photon->deltaXPosition +
                sqrt((1 - mu * mu) / (1 - photon->deltaYPosition * photon->deltaYPosition) / x3) *
                (x1 * photon->deltaXPosition * photon->deltaYPosition + x2 * photon->deltaZPosition);

      	    photon->deltaZPosition = mu *
                                     photon->deltaZPosition +
                                     sqrt((1 - mu * mu) / (1 - photon->deltaYPosition) / x3) *
                                     (x1 * photon->deltaYPosition * photon->deltaZPosition - x2 * photon->deltaXPosition);

            photon->deltaYPosition = mu *
                                     photon->deltaYPosition -
                                     sqrt((1 - mu * mu) * (1 - photon->deltaYPosition * photon->deltaYPosition) / x3) *
                                     x1;

        }

        photon->deltaXPosition = t;
    }
}

/**
 * @brief This function will print the results and statistics of the Monte Carlo simulations.
 * @param rd A statistic used to calculate the results
 * @param bit A statistic used to calculate the results
 * @param heat An array of statistics used to calculate the results
 * @param totalPhotons The total number of generated photons in the simulation
 */

void print_results(double* rd, double* bit, double heat[], long totalPhotons) {

    printf("Small Monte Carlo by Scott Prahl (https://omlc.org)\n");
    printf("1 W/cm^2 Uniform Illumination of Semi-Infinite Medium\n\n");

    printf("Scattering = %8.3f/cm\n", MU_S);
    printf("Absorption = %8.3f/cm\n", MU_A);
    printf("Anisotropy = %8.3f\n", G);
    printf("Refraction Index = %8.3f\n", N);
    printf("Number of Photons = %8ld\n\n", totalPhotons);

    printf("Specular Reflection = %10.5f\n", RS);
    printf("Backscattered Reflection = %10.5f\n\n", (*rd) / ((*bit) + totalPhotons));

    printf("Depth         Heat\n[microns]     [W/cm^3]\n");

    for (int i = 0; i < BINS - 1; i++) {
        printf("%6.0f    %12.5f\n", 
               i * MICRONS_PER_BIN, 
               heat[i] / MICRONS_PER_BIN * 1e4 / ((*bit) + totalPhotons));
    }

    printf("Extra Heat [W/cm^3]  %12.5f\n", heat[BINS - 1] / ((*bit) + totalPhotons));
}

/**
 * @brief Kernel for generating movement with photons 
 * @param d_rd A pointer to an array in global memory. Is a statistic used to calculate the results
 * @param d_bit A pointer to an array in global memory. Is a statistic used to calculate the results
 * @param d_heat A pointer to a 2d array in global memory. Is used to calculate the results
 * @param photons The number of photons being simulated 
 */

__global__ void simulationKernel(double* d_rd, double* d_bit, double* d_heat, int photons) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    double rd = 0.0;
    double bit = 0.0;
    double heat[BINS] = {0};

    Photon photon;

    curandState state;
    curand_init(SEED, i, 0, &state);

    initialize_photon(&photon);

    while(photon.weight > 0) {
        move(&photon, &rd, &state);
        absorb(&photon, &bit, heat, &state);
        scatter(&photon, &state);
    }

    if (i<photons) {
        d_rd[i] += rd;
        d_bit[i] += bit;
        for (int bin = 0; bin < BINS; bin++) {
            d_heat[bin * photons + i] = heat[bin];
        }
    }

}

/**
 * @brief Kernel for reducing all elements in data into grid number of elements
 *   This version uses n/2 threads --
 *   it performs the first level of reduction when reading from global memory.
 *   Heavily inspired by reduce3 - https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu
 * @param data A pointer to an array in global memory 
 * @param n The number of elements in the data array
 */

__global__ void reduceKernel(double* data, int n) {
    extern __shared__ double ds[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    double sum = i < n ? data[i] : 0;

    sum += i + blockDim.x < n ? data[i + blockDim.x] : 0;

    ds[tid] = sum;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum += ds[tid + s];
            ds[tid] = sum;
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        data[blockIdx.x] = sum;
    }

    __syncthreads();
}

/**
 * @brief Calls the reduceKernel for d_rd, d_bit, and d_heat until one value is stored in the first 
 * memory location
 * @param d_rd A pointer to an array in global memory. Is a statistic used to calculate the results
 * @param d_bit A pointer to an array in global memory. Is a statistic used to calculate the results
 * @param d_heat A pointer to a 2d array in global memory. Is used to calculate the results
 * @param offset The starting location in memory for the reduction to take place
 * @param photons The number of photons being simulated 
 * @param totalPhotons The number of photons stored in each array
 */

void reducer(double* d_rd, 
             double* d_bit, 
             double d_heat[], 
             long offset, 
             long photons, 
             long totalPhotons) {
    int grid = ceil(photons / (double) THREADS); 

    reduceKernel<<<grid, THREADS,
        (sizeof(double) * MAXPHOTONS)>>>(&d_rd[offset], photons);
    HANDLE_ERROR(cudaGetLastError());

    reduceKernel<<<grid, THREADS,
        (sizeof(double) * MAXPHOTONS)>>>(&d_bit[offset], photons);
    HANDLE_ERROR(cudaGetLastError());

    // This for loop reduces the heat generated at each depth bin
    for(int bin = 0; bin < BINS; bin++) {
        reduceKernel<<<grid, THREADS,
            (sizeof(double) * MAXPHOTONS)>>>(&d_heat[bin * totalPhotons + offset], photons);
        HANDLE_ERROR(cudaGetLastError());
    }

    // This conditional recursively calls reducer until all photons have been reduced
    if(grid > 1) {
        reducer(d_rd, d_bit, d_heat, offset, grid, totalPhotons);
    }
}

/**
 * @brief Method for starting the simulation and combining the outputs
 * @param h_rd A pointer to an array in host memory. Is a statistic used to calculate the results
 * @param h_bit A pointer to an array in host memory. Is a statistic used to calculate the results
 * @param h_heat A pointer to a 2d array in host memory. Is used to calculate the results
 * @param totalPhotons The number of photons being simulated 
 */

void gpu_simulation(double* h_rd, double* h_bit, double h_heat[], long totalPhotons) {
    double rd = 0.0;
    double bit = 0.0;
    double heat[BINS] = {0};

    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));

    double* d_rd;
    double* d_bit;
    double* d_heat;


    HANDLE_ERROR(cudaMalloc((void**) &d_rd, sizeof(double) * totalPhotons));
    HANDLE_ERROR(cudaMalloc((void**) &d_bit, sizeof(double) * totalPhotons));
    HANDLE_ERROR(cudaMalloc((void**) &d_heat, sizeof(double) * totalPhotons * BINS));

    HANDLE_ERROR(cudaMemset(d_rd, 0, sizeof(double) * totalPhotons));
    HANDLE_ERROR(cudaMemset(d_bit, 0, sizeof(double) * totalPhotons));
    HANDLE_ERROR(cudaMemset(d_heat, 0, sizeof(double) * totalPhotons * BINS));

    simulationKernel<<<ceil(totalPhotons / 512.0), 512>>>(d_rd, d_bit, d_heat, totalPhotons);
    HANDLE_ERROR(cudaGetLastError());

    int photons = totalPhotons;

    // This while loop reduces in chunks of MAXPHOTONS until less than the maximum remain
    while(photons > MAXPHOTONS) {
        photons -= MAXPHOTONS;
        reducer(d_rd, d_bit, d_heat, photons, MAXPHOTONS, totalPhotons);
        photons++;
    }

    reducer(d_rd, d_bit, d_heat, 0, photons, totalPhotons);


    HANDLE_ERROR(cudaMemcpy(&rd, d_rd, sizeof(double), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&bit, d_bit, sizeof(double), cudaMemcpyDeviceToHost));

    // This for loop copies the device heat bins back to the host
    for(int bin = 0; bin < BINS; bin++) {
        HANDLE_ERROR(cudaMemcpy(&heat[bin], 
                                &d_heat[bin * totalPhotons], 
                                sizeof(double), 
                                cudaMemcpyDeviceToHost));
    }

    HANDLE_ERROR(cudaFree(d_rd));
    HANDLE_ERROR(cudaFree(d_bit));
    HANDLE_ERROR(cudaFree(d_heat));

    *h_rd += rd;
    *h_bit += bit;

    for(int bin = 0; bin < BINS; bin++) {
        h_heat[bin] += heat[bin];
    }

}

/*The parallel CUDA code for 2D Ising Model simulation using Metropolis Monte Carlo algorithm
  In this implementation, the random numbers are generated on CPU side.
  When you install the CUDA environment, you can compile the CUDA code in linux terminal directly:
  nvcc ising1.cu -o ising1
  */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// the 2D block size
#define BDIMX 8
#define BDIMY 1

// Monte Carlo sweeps: N Monte Carlo steps - one for each spins, on average
#define sweeps1 6000  
#define sweeps2 3000


// function create initial spins on a lattice
void InitialSpins(int *spins, int N, float msg)
{
    int i;
    float R;
    for (i = 0; i < N; i++)
    {

        R = rand() / (float)(RAND_MAX);
        if (R < msg)
        {
            spins[i] = 1;
        }
        else
        {
            spins[i] = -1;
        }
    }
}



// linSpace Temperature
void linSpaceTemperature(float start, float end, int n, float *Temperature)
{
    int i;
    float h = (end - start) / (n - 1);
    for (i = 0; i < n; i++)
    {
        Temperature[i] = start + i * h;
    }
}


// set the random number generator
void RandGenerator(float *random, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        random[i] = rand() / (float)(RAND_MAX);
    }
}


/* declare global variable on GPU */

// variables for temporarily storing the properties of one step
__device__ int d_m;
__device__ int d_e;

// variables for summing over all the properties of every step
__device__ int d_M;
__device__ int d_E;

// variables for specific heat and magnetic susceptibility
__device__ float d_M2;
__device__ float d_E2;


// calculate the properties
__global__ void CalcProp(int *energy, int *spins, int size)
{
    // map the threads to the global memory
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int nx = blockDim.x * gridDim.x;
    int idx = iy * nx + ix;

    // calculate the properties of the present configuration
    atomicAdd(&d_m, spins[idx]);
    atomicAdd(&d_e, energy[idx]);
    
    if (idx == 0)
    {
        d_M += abs(d_m);
        d_E += d_e;
        d_E2 += (((float)d_e)*d_e)/ (2.0f * 2.0f);
        d_M2 += (((float)d_m)*d_m);
        d_m = 0;
        d_e = 0;
    }
}

// reset the variables after every temperature iteration
__global__ void reset()
{
    // map the threads to the global memory
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int nx = blockDim.x * gridDim.x;
    int idx = iy * nx + ix;

    if (idx == 0)
    {
        d_M = 0;

        d_E = 0;

        d_M2 = 0.;

        d_E2 = 0.;
    }
}

// flip spins using Metropolis algorithm 
__global__ void MetropolisDevice_even(int *spins, int *energy, float *random, const float Beta)
{
    // map the threads to the global memory
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    int idx = iy * nx + ix;

    float rand = random[idx];
    int dE;

    int left, right, up, down;


    // place the value to neighbours with boundary conditions
    if (ix == 0)
    {
        left = spins[idx + nx - 1];
    }
    else
    {
        left = spins[idx - 1];
    }
    if (ix == (ny - 1))
    {
        right = spins[idx - nx + 1];
    }
    else
    {
        right = spins[idx + 1];
    }
    if (iy == 0)
    {
        up = spins[idx + (ny - 1) * nx];
    }

    else
    {
        up = spins[idx - nx];
    }

    if (iy == nx - 1)
    {
        down = spins[idx - (ny - 1) * nx];
    }

    else
    {
        down = spins[idx + nx];
    }

    if ((ix + iy) % 2 == 0) //flip even spins
    {

        energy[idx] = -spins[idx] * (left + right + up + down);
        dE = -2 * energy[idx];

        if (dE < 0 || exp(-dE * Beta) > rand)
        {
            spins[idx] *= -1;
            energy[idx] *= -1;
        }
    }

}

__global__ void MetropolisDevice_odd(int *spins, int *energy, float *random, const float Beta)
{
    // map the threads to the global memory
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = blockDim.x * gridDim.x;
    int ny = blockDim.y * gridDim.y;
    int idx = iy * nx + ix;
    

    float rand = random[idx];
   
    float dE;

    int left, right, up, down;


    // place the value to neighbours with boundary conditions
    if (ix == 0)
    {
        left = spins[idx + nx - 1];
    }
    else
    {
        left = spins[idx - 1];
    }
    if (ix == ny - 1)
    {
        right = spins[idx - nx + 1];
    }
    else
    {
        right = spins[idx + 1];
    }
    if (iy == 0)
    {
        up = spins[idx + (ny - 1) * nx];
    }

    else
    {
        up = spins[idx - nx];
    }

    if (iy == nx - 1)
    {
        down = spins[idx - (ny - 1) * nx];
    }
    else
    {
        down = spins[idx + nx];
    }
    if ((ix + iy) % 2 != 0) //flip odd spins
    {

        energy[idx] = -spins[idx] * (left + right + up + down);
        dE = -2 * (float)energy[idx];

        if (dE < 0 || exp(-dE * Beta) > rand)
        {
            spins[idx] *= -1;
            energy[idx] *= -1;
        }
    }
}

int main()
{
    //latice size
    int size = 8;
    
    printf("CUDA program\n");
    printf("\n%d x %d size latice \n", size, size);
    printf("The random numbers are generated on CPU side\n");

    int i, n; // iteration variables

    float Beta; // beta = 1/T, in this project set k = 1, J = 1.

    // massage to define the initial configuration. setting msg = 0.5 to random configuration. setting msg = 0 to orientated configuration. 
    float msg = 0.5;

    //temperature intervel
    int numberTemperature = 45; // number of temperatures sampled
    float *Temperature = (float*)malloc(numberTemperature * sizeof(float));
    linSpaceTemperature(0.5, 5.0, numberTemperature, Temperature);
    printf("\nTemperature range 0.5 to 5.0\n");

    // averege energy and magnetization per spin
    float *avergEnergy = (float*)malloc(numberTemperature * sizeof(float));
    float *avergMag = (float*)malloc(numberTemperature * sizeof(float));
    
    // variables for calculate specific heat and magnetic susceptibility
    float *avergEnergy2 = (float*)malloc(numberTemperature * sizeof(float));
    float *avergMag2 = (float*)malloc(numberTemperature * sizeof(float));
    
    // specific heat and magnetic susceptibility
    float *heat = (float*)malloc(numberTemperature * sizeof(float));
    float *sus = (float*)malloc(numberTemperature * sizeof(float));

    
    // declare variables and allocate memory
    int *d_spins;
    int *h_spins;
    int *d_energy;
    int *h_energy;
    int *gpuRef; // results return from GPU
    float *h_random_numbers;
    float *d_random_numbers;

    int nxy = size * size;
    int nBytes = nxy * sizeof(int);
    int NBytes = nxy * sizeof(float);

    h_spins = (int *)malloc(nBytes);
    h_energy = (int *)malloc(nBytes);

    gpuRef = (int *)malloc(nBytes);
    h_random_numbers = (float *)malloc(NBytes);
    
    //set random number generator seed
    srand(123456);

    
    // initialize data at host side
    memset(gpuRef, 0, nBytes);
    memset(h_energy, 0, nBytes);
    InitialSpins(h_spins, nxy,msg);


    // malloc device global memory
    cudaMalloc((void **)&d_spins, nBytes);
    cudaMalloc((void **)&d_energy, nBytes);
    cudaMalloc((void **)&d_random_numbers, NBytes);


    // transfer data from host to device
    int h_m = 0;
    int h_e = 0;
    int h_M = 0;
    int h_E = 0;
    float h_M2 = 0.0f;
    float h_E2 = 0.0f;

    cudaMemcpy(d_spins, h_spins, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_energy, h_energy, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_M, &h_M, sizeof(int));
    cudaMemcpyToSymbol(d_E, &h_E, sizeof(int));
    cudaMemcpyToSymbol(d_m, &h_m, sizeof(int));
    cudaMemcpyToSymbol(d_e, &h_e, sizeof(int));

    cudaMemcpyToSymbol(d_M2, &h_M2, sizeof(float));
    cudaMemcpyToSymbol(d_E2, &h_E2, sizeof(float));

    // invoke kernel at host side
    dim3 block(BDIMX, BDIMY);
    dim3 grid(size / BDIMX, size / BDIMY);


    // timing
    clock_t d_start, d_end;
    double d_time_used;
    d_start = clock();
    printf("\nMain loop starting...\n");
    // main loop (loop over the temerature)
    for (n = 0; n < numberTemperature; n++)
    {
        Beta = 1 / Temperature[n];

        // process for equilibrium 
        for (i = 0; i < sweeps1; i++)
        {
            RandGenerator(h_random_numbers, nxy);
            cudaMemcpy(d_random_numbers, h_random_numbers, nBytes, cudaMemcpyHostToDevice);
            MetropolisDevice_even << <grid, block >> > (d_spins, d_energy, d_random_numbers, Beta);
            MetropolisDevice_odd << <grid, block >> > (d_spins, d_energy, d_random_numbers, Beta);
        }

        // process for calculating the properties
        for (i = 0; i < sweeps2; i++)
        {
            RandGenerator(h_random_numbers, nxy);
            cudaMemcpy(d_random_numbers, h_random_numbers, nBytes, cudaMemcpyHostToDevice);
            MetropolisDevice_even << <grid, block >> > (d_spins, d_energy, d_random_numbers, Beta);

            MetropolisDevice_odd << <grid, block >> > (d_spins, d_energy, d_random_numbers, Beta);

           
                //printf("Temperature %.3f Iteration %d\n", Temperature[n], i + 1);
                CalcProp <<<grid, block >>> (d_energy, d_spins,size);
                //cudaDeviceSynchronize();
            
        }

        cudaMemcpyFromSymbol(&h_M, d_M, sizeof(int));
        
        cudaMemcpyFromSymbol(&h_E, d_E, sizeof(int));

        cudaMemcpyFromSymbol(&h_M2, d_M2, sizeof(float));

        cudaMemcpyFromSymbol(&h_E2, d_E2, sizeof(float));
        
        // calculate the average propeties per spin
        avergEnergy[n] = h_E / ((sweeps2 )*((float)(size*size))*2.0f);
        avergMag[n] = h_M / ((sweeps2 )*((float)(size*size)));
        
        avergEnergy2[n] = h_E2 / ((sweeps2 ));
        avergMag2[n] = h_M2 / ((sweeps2));

        heat[n] = (avergEnergy2[n]/ ((float)(size*size)) - avergEnergy[n] * avergEnergy[n]*((size*size)))*Beta*Beta;
        sus[n] = (avergMag2[n]/ ((float)(size*size)) - avergMag[n] * avergMag[n]*(size*size))*Beta;
        
        reset << <grid, block >> > ();
    }

    d_end = clock();

    cudaMemcpy(gpuRef, d_spins, nBytes, cudaMemcpyDeviceToHost);
    
    d_time_used = ((double)(d_end - d_start)) / CLOCKS_PER_SEC;

    printf("\nEnd main loop.\nTotal time using GPU %f s\n", d_time_used);
    
    // deallocate the GPU memory
    cudaFree(d_random_numbers);
    cudaFree(d_spins);
    cudaFree(d_energy);
    cudaDeviceReset();

    
    FILE *properties;
    properties = fopen("Properties_CUDA1.txt", "a");
    fprintf(properties, "%d x %d size lattice\n", size, size);
    fprintf(properties, "\nTemperature  Energy  Magnetization  Specific heat  Magnetic susceptibility (per spin)\n");
    for (i = 0; i < numberTemperature; i++)
    {
        fprintf(properties, "%.2f         %.3f  %.3f          %.3f          %.3f \n", Temperature[i], avergEnergy[i], \
            avergMag[i], heat[i], sus[i]);
    }
    fclose(properties);

    // print out the properties
    printf("\nTemperature  Energy  Magnetization  Specific heat  Magnetic susceptibility (per spin)\n");
    for (i = 0; i < numberTemperature; i++)
    {
        printf("%.2f         %.3f  %.3f          %.3f          %.3f \n", \
            Temperature[i], avergEnergy[i], \
            avergMag[i], heat[i], sus[i]);
    }

    // deallocate the memory
    free(h_spins);
    free(h_random_numbers);
    free(Temperature);
    free(h_energy);


    printf("end\n");
    return 0;
}


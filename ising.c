/*The serial code for 2D Ising Model simulation using Metropolis Monte Carlo algorithm 
  You can compile the C code in linux terminal directly:
   gcc -o ising ising.c -lm 
   */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>


// function create initial spins on a lattice
// a message is passed to define the initial configuration.
void InitialSpins(int **spins, int size, double msg)
{
	int i, j;
	double R;
  for (j = 0; j < size; j++)
  {
      for (i = 0; i < size; i++)
      {
          R = rand() / (double)(RAND_MAX);
          if (R < msg)
          {
              spins[i][j] = 1;
          }
          else
          {
              spins[i][j] = -1;
          }
      }
  }
}


// flip spins using Metropolis algorithm
void Metropolis(int **spins, int size, double Beta)
{
	int k, i, j;
	int Oldspin, Newspin;
	int dE;
  double r;
  int left, right, up, down; // nearest neighbouring spins
  for (k = 0; k < size * size; k++)
  {
      i = rand() % size;
      j = rand() % size;
      Oldspin = spins[i][j];
      Newspin = -Oldspin;

      // periodic boundary conditions
      left = spins[(i + size - 1) % size][j];
      right = spins[(i + 1) % size][j];
      up = spins[i][(j + size - 1) % size];
      down = spins[i][(j + 1) % size];

      dE = -(Newspin - Oldspin) * (left + right + up + down);
      r = rand() / (double)(RAND_MAX);
      if (dE < 0 || exp(-dE * Beta) > r)
      {
          spins[i][j] = Newspin;
      }
  }
 }



// potential energy
int Potential_Energy(int **spins, int size)
{
	int i, j;
	int energy = 0;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			energy += -spins[i][j] * (spins[(i + 1) % size][j] + spins[(i +size- 1) % size][j] + spins[i][(j + 1) % size] + spins[i][(j + size - 1) % size]);
		}
	}
	return energy;
}

// absolute magnetization
int Magnetization(int **spins, int size)
{
	int i, j;
	int Mag = 0;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			Mag += spins[i][j];
		}
	}
	return abs(Mag);
}


// linSpace Temperature
void linSpaceTemperature(double start, double end, int n, double *Temperature)
{
	int i;
	double h = (end - start) / (n - 1);
	for (i = 0; i < n; i++)
	{
      Temperature[i] = start + i * h;
	}
}

int main()
{

  //latice size
  int size = 8;

  printf("C program\n");
  printf("\n%d x %d size latice \n", size, size);

  int N; // total number of spins
  N = size * size;

  // iteration variables
  int i, j, n;

  // 2D spins array
  int **spins;
  spins = (int **)malloc(size * sizeof(int*));
  for (i = 0; i < size;i++) {
      spins[i] = (int*)malloc(size * sizeof(int));
  }

  //temperature intervel
  int numberTemperature = 45; // number of temperatures sampled
	double *Temperature = (double*)malloc(numberTemperature * sizeof(double));
	linSpaceTemperature(0.5, 5.0, numberTemperature, Temperature);
  
  printf("\nTemperature range 0.5 to 5.0\n");

  // massage to define the initial configuration. setting msg = 0.5 to random configuration. setting msg = 0 to orientated configuration. 
  double msg = 0.5; 

  /*Monte Carlo sweeps: N Monte Carlo steps - one for each spins, on average*/
	int sweeps1 = 6000;
  int sweeps2 = 3000;
  printf("\nnumber of iterations for equilubrim: %d\n", sweeps1);
  printf("number of iterations for sampling: %d\n", sweeps2);

  double Beta; // beta = 1/T, in this project set k = 1, J = 1.

  // energy and magnetization 
  int *Energy = (int*)malloc(numberTemperature * sizeof(int));
  int *Mag = (int*)malloc(numberTemperature * sizeof(int));
  
  // initialize energy and magnetization
  memset(Energy, 0, numberTemperature * sizeof(int));
  memset(Mag, 0, numberTemperature * sizeof(int));

  // averege energy and magnetization per spin
  double *avergEnergy = (double*)malloc(numberTemperature * sizeof(double));
  double *avergMag = (double*)malloc(numberTemperature * sizeof(double));

  // variables for calculate specific heat and magnetic susceptibility
  double *Mag2 = (double*)malloc(numberTemperature * sizeof(double));
  double *Energy2 = (double*)malloc(numberTemperature * sizeof(double));

  for (j = 0; j < numberTemperature; j++)
  {
      Mag2[j] = 0.0;
  }

  for (j = 0; j < numberTemperature; j++)
  {
      Energy2[j] = 0.0;
  }

  double *avergMag2 = (double*)malloc(numberTemperature * sizeof(double));
  double *avergEnergy2 = (double*)malloc(numberTemperature * sizeof(double));

  // specific heat and magnetic susceptibility
  double *heat = (double*)malloc(numberTemperature * sizeof(double));
  double *sus = (double*)malloc(numberTemperature * sizeof(double));

  // set random number seeds
  srand(123456);

  // timing
  clock_t start, end;
  double time_used;
  start = clock();
  printf("\nMain loop starting...\n");
  // main loop (loop over the temerature)
  for (n = 0; n < numberTemperature; n++)
   {
      // initalize the latice
      InitialSpins(spins, size,msg);

      Beta = 1 / Temperature[n];

      // process for equilibrium 
      for (i = 0; i < sweeps1; i++)
      {
          //fliping the spins using metropolis algorithm  
          Metropolis(spins, size, Beta);
      }
      
      // process for calculating the properties
      for (j = 0; j < sweeps2; j++)
      {
          // fliping the spins using metropolis algorithm  
          Metropolis(spins, size, Beta);
          // sample the properties  
          Energy[n] += Potential_Energy(spins, size);
          Mag[n] += Magnetization(spins, size);
          Energy2[n] += (double)(Potential_Energy(spins, size) * Potential_Energy(spins, size));
          Mag2[n] += (double)(Magnetization(spins, size)* Magnetization(spins, size));;
  
      }

      // calculate the average propeties per spin
      avergEnergy[n] = Energy[n] /((sweeps2)* ((double)(N))*2); // energy be calculated twice when sum over spins
      avergMag[n] = Mag[n] / ((sweeps2)*((double)(N)));
      
      avergEnergy2[n] = Energy2[n] / (sweeps2 *(2 * 2)); // Energy2 is the double of the Energy
      avergMag2[n] = Mag2[n] / sweeps2;
      
      
      heat[n] = (avergEnergy2[n] - avergEnergy[n] * avergEnergy[n]*N * N)*Beta*Beta/N;
      sus[n] = (avergMag2[n] - avergMag[n] * avergMag[n] * N * N)*Beta/N;

  }
  
  end = clock();

  time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  
  printf("\nEnd main loop.\nTotal time %f s\n", time_used);


  
  FILE *properties;
  properties = fopen("Properties_C.txt", "a");
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
  free(spins);
  free(Temperature);
  free(Energy);
  free(Mag);
  free(Energy2);
  free(Mag2);
  free(avergEnergy);
  free(avergMag);
  free(avergEnergy2);
  free(avergMag2);
  free(heat);
  free(sus);

  printf("end\n");

  return 0;
}





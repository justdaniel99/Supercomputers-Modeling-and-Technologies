#define _USE_MATH_DEFINES

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int N = 10000; // default dots number
const int masterRoot = 0;

const double exactValue = M_PI_2 * M_E - M_PI;
const double a1 = -1.0, b1 = 1.0; 
const double a2 = -1.0, b2 = 1.0; 
const double a3 = 0.0, b3 = 1.0; 
const double volume = (b1 - a1) * (b2 - a2) * (b3 - a3);

int rank = 0; // default rank
int processesNumber = 2; // default processes number

double F(double x, double y, double z)
{
    if (x * x + y * y + z * z <= 1)
    {
        return exp(x * x + y * y) * z;
    }

    return 0.0;
}

void masterProcess(double eps, int* displs, int* sendcounts, double* dotsCoordinates, int* randomCoefficient)
{
    int receiveBufferScatter, dotsNumber = 0;
    int dotsDistributionBetweenWorkers = N / (processesNumber - 1);
    int undistribtedDots = N % (processesNumber - 1);

    double error, executionTime, totalSum = 0.0, approximateValue = 0.0, masterTime = MPI_Wtime();

    displs[0] = 0;
    sendcounts[0] = 0;

    for (int i = 1; i < processesNumber; i++)
    {
        int undistribtedDotsDistribution = 0;

        if (i - 1 < undistribtedDots)
        {
            undistribtedDotsDistribution = 1;
        }

        sendcounts[i] = 3 * (dotsDistributionBetweenWorkers + undistribtedDotsDistribution);
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }

    do
    {
        srand(*randomCoefficient); // initialization of the random number generator rand() // version 1
        *randomCoefficient += 1;

        for (int i = 0; i < N; i++)
        {
            dotsCoordinates[3 * i] = a1 + (b1 - a1) * ((double)rand() / (double)RAND_MAX); 
            dotsCoordinates[3 * i + 1] = a2 + (b2 - a2) * ((double)rand() / (double)RAND_MAX);
            dotsCoordinates[3 * i + 2] = a3 + (b3 - a3) * ((double)rand() / (double)RAND_MAX);
        }

        dotsNumber += N;
            
        MPI_Scatter(sendcounts, 1, MPI_INT, &receiveBufferScatter, 1, MPI_INT, masterRoot, MPI_COMM_WORLD); // sending signals for workers to start

        MPI_Scatterv(dotsCoordinates, sendcounts, displs, MPI_DOUBLE, MPI_IN_PLACE, 0, MPI_DOUBLE, masterRoot, MPI_COMM_WORLD); // sending dots subsequence to workers

        double receivedSum = 0.0, sendBufferReduce = 0.0;
        MPI_Reduce(&sendBufferReduce, &receivedSum, 1, MPI_DOUBLE, MPI_SUM, masterRoot, MPI_COMM_WORLD); // receiving sum from workers
        totalSum += receivedSum;

        approximateValue = volume * totalSum / dotsNumber;
        error = fabs(exactValue - approximateValue);

    } while (error >= eps);

    for (int i = 1; i < processesNumber; ++i)
    {
        sendcounts[i] = 0; // making signals for workers to finish
    }

    MPI_Scatter(sendcounts, 1, MPI_INT, &receiveBufferScatter, 1, MPI_INT, masterRoot, MPI_COMM_WORLD); // sending signals for workers to finish

    masterTime = MPI_Wtime() - masterTime;
    MPI_Reduce(&masterTime, &executionTime, 1, MPI_DOUBLE, MPI_MAX, masterRoot, MPI_COMM_WORLD);

    printf("Calculated approximate integral value:   %f\n", approximateValue);
    printf("Calculated value error:      %f\n", error);
    printf("Generated random dots number:    %d\n", dotsNumber);
    printf("Program execution time: %f\n\n", executionTime);
}

void workerCalculation(double* dotsCoordinates)
{
    int receiveBufferScatter;
    double workerSum, receiveBufferReduce, totalTime, workerTimer = MPI_Wtime();

    MPI_Scatter(NULL, 0, MPI_INT, &receiveBufferScatter, 1, MPI_INT, masterRoot, MPI_COMM_WORLD); // receiving signals from master to start

    do
    {
        MPI_Scatterv(NULL, NULL, NULL, MPI_DOUBLE, dotsCoordinates, receiveBufferScatter, MPI_DOUBLE, masterRoot, MPI_COMM_WORLD); // receiving dots from master

        workerSum = 0.0;
        for (int i = 0; i < receiveBufferScatter / 3; ++i)
        {
            workerSum += F(dotsCoordinates[3 * i], dotsCoordinates[3 * i + 1], dotsCoordinates[3 * i + 2]);
        }
            
        MPI_Reduce(&workerSum, &receiveBufferReduce, 1, MPI_DOUBLE, MPI_SUM, masterRoot, MPI_COMM_WORLD); // sending sum to master
        MPI_Scatter(NULL, 0, MPI_INT, &receiveBufferScatter, 1, MPI_INT, masterRoot, MPI_COMM_WORLD); // waiting signals from master to finish

    } while (receiveBufferScatter > 0);

    workerTimer = MPI_Wtime() - workerTimer;
    MPI_Reduce(&workerTimer, &totalTime, 1, MPI_DOUBLE, MPI_MAX, masterRoot, MPI_COMM_WORLD);
}

int main(int argc, char** argv)
{

    //srand(time(NULL)); // initialization of the random number generator rand() // version 2

    double eps = atof(argv[1]); // reading epsilon value from command line and converting from string to double

    MPI_Init(&argc, &argv); // initialization function
    MPI_Comm_size(MPI_COMM_WORLD, &processesNumber); // processes number detection function
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // process number detection function
    
    if (rank == 0) // master
    {
        double* dotsCoordinates = new double[3 * N]; 
        int* displs = new int[processesNumber];
        int* sendcounts = new int[processesNumber]; 
        int randomCoefficient = 0;

        masterProcess(eps, displs, sendcounts, dotsCoordinates, &randomCoefficient);

        delete[] dotsCoordinates;
        delete[] displs;
        delete[] sendcounts;
    }
    else // worker
    { 
        double* dots = new double[3 * N / (processesNumber - 1) + 3];

        workerCalculation(dots);
        
        delete[] dots;

    }
    
    MPI_Finalize(); // completion function
    
    return 0;
}




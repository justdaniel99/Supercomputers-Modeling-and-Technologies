#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

const int masterRoot = 0;

// rectangle boundaries
const double A1 = -1.0;
const double A2 = 2.0;
const double B1 = -2.0;
const double B2 = 2.0;

const double eps = 1e-4; // iteration method accuracy

int M, N; //grid sizes by X and Y
int xProcesses, yProcesses; //number of processes by X and Y

int rank = 0; // default rank
int processesNumber = 1; // default processes number

// processes distribution between X and Y axes
int processesDistribution()
{
	int result = 1;
	int min = abs((M + 1) - N / processesNumber);
	int current = abs((M + 1) / processesNumber - N);

	if (current < min)
	{
		result = processesNumber;
		min = current;
	}

	for (int i = 2; i * i <= processesNumber; ++i)
	{
		if (processesNumber % i == 0)
		{
			int j = processesNumber / i;

			current = abs((M + 1) / i - N / j);
			if (current < min)
			{
				result = i;
				min = current;
			}

			current = abs((M + 1) / j - N / i);
			if (current < min)
			{
				result = j;
				min = current;
			}
		}
	}

	return result;
}

// potential
double q(double x, double y)
{
	return (x + y) * (x + y);
}

// exact function
double u(double x, double y)
{
	return exp(1.0 - q(x, y));
}

//Poisson equation koefficient (in Laplace operator)
double k(double x)
{
	return 4.0 + x;
}

// F
double F(double x, double y)
{
	return (-1) * u(x, y) * (8 * x * x * x + 16 * x * x * y + 31 * x * x + 8 * x * y * y + 62 * x * y - 6 * x + 31 * y * y - 2 * y - 16);
}

// left boundary - 2nd type condition
double psiL(double y) {
	return 2.0 * (4.0 + A1) * (A1 + y) * u(A1, y);
}

// right boundary - 2nd type condition
double psiR(double y) {
	return (-2.0) * (4.0 + A2) * (A2 + y) * u(A2, y);
}

// bottom boundary - 1st type condition
double phi(double x) {
	return u(x, B1);
}

// upper boundary - 2nd type condition
double psiT(double x) {
	return -2.0 * (4.0 + x) * (x + B2) * u(x, B2);
}

// grid coefficient "a"
double a(int i, double hx)
{
	double res = k(-1.0 + i * hx - 0.5 * hx);
	return res;
}

// grid coefficient "b"
double b(int i, double hx)
{
	double res = k(-1.0 + i * hx);
	return res;
}

// left difference derivative of function w by x
double leftDifDefWX(double* w, int i, int j, double hx)
{
	double res = w[i + j * (M + 1)] - w[i - 1 + j * (M + 1)];
	return res / hx;
}

// left difference derivative of function w by y
double leftDifDefWY(double* w, int i, int j, double hy)
{
	double res = w[i + j * (M + 1)] - w[i + (j - 1) * (M + 1)];
	return res / hy;
}

// right difference derivative of function a*w_{x,i-1,j} by x
double rightDifDefAWX(double* w, int i, int j, double hx)
{
	double res = a(i + 1, hx) * leftDifDefWX(w, i + 1, j, hx) - a(i, hx) * leftDifDefWX(w, i, j, hx);
	return res / hx;
}

// right difference derivative of function b*w_{y,i,j-1} by y
double rightDifDefBWY(double* w, int i, int j, double hx, double hy)
{
	double res = b(i, hx) * leftDifDefWY(w, i, j + 1, hy) - b(i, hx) * leftDifDefWY(w, i, j, hy);
	return res / hy;
}

// Laplas difference operator
double laplasDifOperator(double* w, int i, int j, double hx, double hy)
{
	double res = rightDifDefAWX(w, i, j, hx) + rightDifDefBWY(w, i, j, hx, hy);
	return res;
}

// boundary value problem approximation
double mainDiffEquation(double* w, int i, int j, double hx, double hy)
{
	double res = q(-1.0 + i * hx, -2.0 + j * hy) * w[i + j * (M + 1)] - laplasDifOperator(w, i, j, hx, hy);
	return res;
}

double leftBorderApproximation(double* w, int i, int j, double hx, double hy)
{
	double res = (-2.0) * a(i + 1, hx) * leftDifDefWX(w, i + 1, j, hx) / hx + q(-1.0 + i * hx, -2.0 + j * hy) * w[i + j * (M + 1)] - rightDifDefBWY(w, i, j, hx, hy);
	return res;
}

double rightBorderApproximation(double* w, int i, int j, double hx, double hy)
{
	double res = 2.0 * a(i, hx) * leftDifDefWX(w, i, j, hx) / hx + q(-1.0 + i * hx, -2.0 + j * hy) * w[i + j * (M + 1)] - rightDifDefBWY(w, i, j, hx, hy);
	return res;
}

double upperBorderApproximation(double* w, int i, int j, double hx, double hy)
{
	double res = 2.0 * b(i, hx) * leftDifDefWY(w, i, j, hx) / hy + q(-1.0 + i * hx, -2.0 + j * hy) * w[i + j * (M + 1)] - rightDifDefAWX(w, i, j, hx);
	return res;
}

double lowerBorderApproximation(double* w, int i, int j, double hx, double hy)
{
	double res = (-1.0) * rightDifDefAWX(w, i, j, hx) + q(-1.0 + i * hx, -2.0 + j * hy) * w[i + j * (M + 1)] - (b(i, hx) * leftDifDefWY(w, i, j + 1, hy) - (b(i, hx) * w[i + j * (M + 1)]) / hy) / hy;
	return res;
}

// left part of "Aw=B" system
void diffSchemeA(double* Aw, double* w, int domainLeftX, int domainRightX, int domainLowerY, int domainUpperY,
	int rectangleLeftX, int rectangleRightX, int rectangleLowerY, int rectangleUpperY,
	int leftPresence, int rightPresence, int upperPresence, int lowerPresence, double hx, double hy)
{
	//#pragma omp parallel for // uncomment for hybrid version of program
	for (int i = domainLeftX; i <= domainRightX; i++)
	{
		for (int j = domainLowerY; j <= domainUpperY; j++)
		{
			Aw[i + j * (M + 1)] = mainDiffEquation(w, i, j, hx, hy);
		}
	}

	// left border
	if (!leftPresence) {
		for (int j = domainLowerY; j <= domainUpperY; j++)
		{
			Aw[rectangleLeftX + j * (M + 1)] = leftBorderApproximation(w, 0, j, hx, hy);
		}
	}

	// right border
	if (!rightPresence) {
		for (int j = domainLowerY; j <= domainUpperY; j++)
		{
			Aw[(rectangleRightX - 1) + j * (M + 1)] = rightBorderApproximation(w, M, j, hx, hy);
		}
	}

	// upper border
	if (!upperPresence) {
		for (int i = domainLeftX; i <= domainRightX; i++)
		{
			Aw[i + (rectangleUpperY - 1) * (M + 1)] = upperBorderApproximation(w, i, N, hx, hy);
		}
	}

	// lower border
	if (!lowerPresence) {
		for (int i = domainLeftX; i <= domainRightX; i++)
		{
			Aw[i + (rectangleLowerY + 1) * (M + 1)] = lowerBorderApproximation(w, i, 1, hx, hy);
		}
	}

	// corners
	if (!upperPresence && !leftPresence)
	{
		Aw[rectangleLeftX + (rectangleUpperY - 1) * (M + 1)] = (-2.0) * a(1, hx) * leftDifDefWX(w, 1, N, hx) / hx + 2.0 * b(0, hx) * leftDifDefWY(w, 0, N, hy) / hy + q(-1.0, -2.0 + N * hy) * w[N * (M + 1)];
	}

	if (!upperPresence && !rightPresence)
	{
		Aw[(rectangleRightX - 1) + (rectangleUpperY - 1) * (M + 1)] = 2.0 * a(M, hx) * leftDifDefWX(w, M, N, hx) / hx + 2.0 * b(M, hx) * leftDifDefWY(w, M, N, hy) / hy + q(-1.0 + M * hx, -2.0 + N * hy) * w[M + N * (M + 1)];
	}

	if (!lowerPresence && !leftPresence)
	{
		Aw[rectangleLeftX + (rectangleLowerY + 1) * (M + 1)] = (-2.0) * a(1, hx) * leftDifDefWX(w, 1, 1, hx) / hx - (b(0, hx) * leftDifDefWY(w, 0, 2, hx) - b(0, hx) * w[M + 1] / hy) / hy + q(-1.0, -2.0 + hy) * w[M + 1];
	}

	if (!lowerPresence && !rightPresence)
	{
		Aw[(rectangleRightX - 1) + (rectangleLowerY + 1) * (M + 1)] = 2.0 * a(M, hx) * leftDifDefWX(w, M, 1, hx) / hx - (b(M, hx) * leftDifDefWY(w, M, 2, hx) - b(M, hx) * w[M + (M + 1)] / hy) / hy + q(-1.0 + M * hx, -2.0 + hy) * w[M + (M + 1)];
	}
}

// right part of "Aw=B" system
void diffSchemeB(double* B, int M, int N, double hx, double hy)
{
	for (int i = 0; i < M + 1; i++)
	{
		for (int j = 0; j < N + 1; j++) {
			B[i + j * (M + 1)] = F(-1.0 + i * hx, -2.0 + j * hy);
		}
	}

	// left border
	for (int j = 0; j < N + 1; j++)
	{
		B[j * (M + 1)] = F(-1.0, -2.0 + j * hy) + 2.0 * psiL(-2.0 + j * hy) / hx;
	}

	// right border
	for (int j = 0; j < N + 1; j++)
	{
		B[M + j * (M + 1)] = F(-1.0 + M * hx, -2.0 + j * hy) + 2.0 * psiR(-2.0 + j * hy) / hx;
	}

	// upper border
	for (int i = 0; i < M + 1; i++)
	{
		B[i + N * (M + 1)] = F(-1.0 + i * hx, -2.0 + N * hy) + 2.0 * psiT(-1.0 + i * hx) / hy;
	}

	// lower border
	for (int i = 0; i < M + 1; i++)
	{
		B[i + (M + 1)] = F(-1.0 + i * hx, -2.0 + hy) + b(i, hx) * phi(-1.0 + i * hx) / (hy * hy);
		B[i] = phi(-1.0 + i * hx);
	}

	// corners
	B[N * (M + 1)] = F(-1.0, -2.0 + N * hy) + 2.0 * psiL(-2.0 + N * hy) / hx + 2.0 * psiT(-1.0) / hy;
	B[M + N * (M + 1)] = F(-1.0 + M * hx, -2.0 + N * hy) + 2.0 * psiL(-2.0 + N * hy) / hx + 2.0 * psiT(-1.0 + M * hx) / hy;
	B[(M + 1)] = F(-1.0, -2.0 + hy) + 2.0 * psiL(-2.0 + hy) / hx + b(0, hx) * phi(-1.0) / (hy * hy);
	B[M + (M + 1)] = F(-1.0 + M * hx, -2.0 + hy) + 2.0 * psiR(-2.0 + hy) / hx + b(M, hx) * phi(-1.0 + M * hx) / (hy * hy);
}

// dots exchange with nearby domains
void exchange(double* w, int domainLeftX, int domainRightX, int domainLowerY, int domainUpperY,
	int leftPresence, int rightPresence, int upperPresence, int lowerPresence,
	int rank, int xProcesses, int yProcesses, MPI_Datatype type)
{
	MPI_Request upperRequest;
	MPI_Request lowerRequest;
	MPI_Request leftRequest;
	MPI_Request rightRequest;

	MPI_Status upperStatus;
	MPI_Status lowerStatus;
	MPI_Status leftStatus;
	MPI_Status rightStatus;

	// send
	if (upperPresence)
	{
		MPI_Isend(w + domainLeftX + domainUpperY * (M + 1), domainRightX - domainLeftX + 1, MPI_DOUBLE, rank + xProcesses, 0, MPI_COMM_WORLD, &upperRequest);
	}

	if (lowerPresence)
	{
		MPI_Isend(w + domainLeftX + domainLowerY * (M + 1), domainRightX - domainLeftX + 1, MPI_DOUBLE, rank - xProcesses, 0, MPI_COMM_WORLD, &lowerRequest);
	}

	if (leftPresence)
	{
		MPI_Isend(w + domainLeftX + domainLowerY * (M + 1), 1, type, rank - 1, 0, MPI_COMM_WORLD, &leftRequest);
	}

	if (rightPresence)
	{
		MPI_Isend(w + domainRightX + domainLowerY * (M + 1), 1, type, rank + 1, 0, MPI_COMM_WORLD, &rightRequest);
	}

	// receive
	if (upperPresence)
	{
		MPI_Recv(w + domainLeftX + (domainUpperY + 1) * (M + 1), domainRightX - domainLeftX + 1, MPI_DOUBLE, rank + xProcesses, 0, MPI_COMM_WORLD, &upperStatus);
	}

	if (lowerPresence)
	{
		MPI_Recv(w + domainLeftX + (domainLowerY - 1) * (M + 1), domainRightX - domainLeftX + 1, MPI_DOUBLE, rank - xProcesses, 0, MPI_COMM_WORLD, &lowerStatus);
	}

	if (leftPresence)
	{
		MPI_Recv(w + domainLeftX - 1 + domainLowerY * (M + 1), 1, type, rank - 1, 0, MPI_COMM_WORLD, &leftStatus);
	}

	if (rightPresence)
	{
		MPI_Recv(w + domainRightX + 1 + domainLowerY * (M + 1), 1, type, rank + 1, 0, MPI_COMM_WORLD, &rightStatus);
	}
}

// weight functions
double rhoX(int i, int M, int N)
{
	if (i == 0 || i == M)
	{
		return 0.5;
	}
	else
	{
		return 1.0;
	}
}

double rhoY(int j, int M, int N)
{
	if (j == 0 || j == N)
	{
		return 0.5;
	}
	else
	{
	return 1.0;
	}
}

void borderInitialization(double* previousW, double* nextW, double hx, double hy)
{
	for (int i = 0; i < (M + 1); i++)
	{
		previousW[i] = phi(A1 + i * hx);
		nextW[i] = phi(A1 + i * hx);
	}
}

int main(int argc, char** argv)
{
	M = atoi(argv[1]);
	N = atoi(argv[2]);

	double hx = (A2 - A1) / (double(M)); // grid steps by X
	double hy = (B2 - B1) / (double(N)); // grid steps by Y

	double tau;
	int iterations = 0;
	int accuracy = 1;

	double* previousW = new double[(M + 1) * (N + 1)];
	double* nextW = new double[(M + 1) * (N + 1)];
	double* r = new double[(M + 1) * (N + 1)];
	double* Ar = new double[(M + 1) * (N + 1)];
	double* Aw = new double[(M + 1) * (N + 1)];
	double* B = new double[(M + 1) * (N + 1)];
	double* difference = new double[(M + 1) * (N + 1)];

	for (int i = 0; i < (M + 1) * (N + 1); i++)
	{
		previousW[i] = 0.0;
		nextW[i] = 0.0;
		r[i] = 0.0;
		Ar[i] = 0.0;
		Aw[i] = 0.0;
		B[i] = 0.0;
		difference[i] = 0.0;
	}

	borderInitialization(previousW, nextW, hx, hy);

	diffSchemeB(B, M, N, hx, hy);

	MPI_Init(&argc, &argv); // initialization function
	MPI_Comm_size(MPI_COMM_WORLD, &processesNumber); // processes number detection function
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // process number detection function

	// begin time
	double wtime = MPI_Wtime();
	double overall_wtime;

	xProcesses = processesDistribution(); // the number of processes on X axis
	yProcesses = processesNumber / xProcesses; // the number of processes on Y axis

	int xDomain = rank % xProcesses;
	int yDomain = rank / xProcesses;

	int rectangleLeftX = xDomain * ((M + 1) / xProcesses);
	int rectangleRightX = (xDomain + 1) * ((M + 1) / xProcesses);
	int rectangleLowerY = yDomain * ((N + 1) / yProcesses);
	int rectangleUpperY = (yDomain + 1) * ((N + 1) / yProcesses);

	if ((M + 1) % xProcesses)
	{
		if (xDomain < ((M + 1) % xProcesses))
		{
			rectangleLeftX += xDomain;
			rectangleRightX += 1 + xDomain;
		}
		else
		{
			rectangleLeftX += xDomain;
			rectangleRightX += xDomain;
		}
	}

	if ((N + 1) % yProcesses)
	{
		if (yDomain < ((N + 1) % yProcesses))
		{
			rectangleUpperY += 1 + yDomain;
			rectangleLowerY += yDomain;
		}
		else
		{
			rectangleUpperY += yDomain;
			rectangleLowerY += yDomain;
		}
	}

	int leftPresence = 0;
	int rightPresence = 0;
	int upperPresence = 0;
	int lowerPresence = 0;

	if (!(rank % xProcesses == 0))
	{
		leftPresence = 1;
	}
	if (!(rank % xProcesses == xProcesses - 1))
	{
		rightPresence = 1;
	}
	if (!(rank >= xProcesses * (yProcesses - 1)))
	{
		upperPresence = 1;
	}
	if (!(rank < xProcesses))
	{
		lowerPresence = 1;
	}

	int domainLeftX = -1;
	int domainRightX = -1;
	int domainLowerY = -1;
	int domainUpperY = -1;

	if (!rightPresence)
	{
		domainRightX = rectangleRightX - 2;
	}
	else
	{
		domainRightX = rectangleRightX - 1;
	}

	if (!leftPresence)
	{
		domainLeftX = rectangleLeftX + 1;
	}
	else
	{
		domainLeftX = rectangleLeftX;
	}

	if (!upperPresence)
	{
		domainUpperY = rectangleUpperY - 2;
	}
	else
	{
		domainUpperY = rectangleUpperY - 1;
	}

	if (!lowerPresence)
	{
		domainLowerY = rectangleLowerY + 1;
	}
	else
	{
		domainLowerY = rectangleLowerY;
	}

	MPI_Datatype leftRightPresenceDataType;
	MPI_Type_vector(domainUpperY - domainLowerY + 1, 1, M + 1, MPI_DOUBLE, &leftRightPresenceDataType);
	MPI_Type_commit(&leftRightPresenceDataType);

	while (accuracy == 1)
	{
		iterations++;

		exchange(previousW, domainLeftX, domainRightX, domainLowerY, domainUpperY,
			leftPresence, rightPresence, upperPresence, lowerPresence,
			rank, xProcesses, yProcesses, leftRightPresenceDataType);

		diffSchemeA(Aw, previousW, domainLeftX, domainRightX, domainLowerY, domainUpperY,
			rectangleLeftX, rectangleRightX, rectangleLowerY, rectangleUpperY,
			leftPresence, rightPresence, upperPresence, lowerPresence, hx, hy);
		
		//#pragma omp parallel for //
		for (int i = domainLeftX; i <= domainRightX; i++)
		{
			for (int j = domainLowerY; j <= domainUpperY; j++)
			{
				r[i + j * (M + 1)] = Aw[i + j * (M + 1)] - B[i + j * (M + 1)];
			}
		}

		exchange(r, domainLeftX, domainRightX, domainLowerY, domainUpperY,
			leftPresence, rightPresence, upperPresence, lowerPresence,
			rank, xProcesses, yProcesses, leftRightPresenceDataType);

		diffSchemeA(Ar, r, domainLeftX, domainRightX, domainLowerY, domainUpperY,
			rectangleLeftX, rectangleRightX, rectangleLowerY, rectangleUpperY,
			leftPresence, rightPresence, upperPresence, lowerPresence, hx, hy);

		double tauNumerator = 0.0;
		double totalTauNumerator = 0.0;

		for (int i = domainLeftX; i <= domainRightX; i++)
		{
			for (int j = domainLowerY; j <= domainUpperY; j++)
			{
				tauNumerator += hx * hy * Ar[i + j * (M + 1)] * r[i + j * (M + 1)] * rhoX(i, M, N) * rhoY(j, M, N);
			}
		}
		MPI_Allreduce(&tauNumerator, &totalTauNumerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		double tauNominator = 0.0;
		double totalTauNominator = 0.0;

		for (int i = domainLeftX; i <= domainRightX; i++)
		{
			for (int j = domainLowerY; j <= domainUpperY; j++)
			{
				tauNominator += hx * hy * Ar[i + j * (M + 1)] * Ar[i + j * (M + 1)] * rhoX(i, M, N) * rhoY(j, M, N);
			}
		}
		MPI_Allreduce(&tauNominator, &totalTauNominator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		tau = totalTauNumerator / totalTauNominator;

		for (int i = domainLeftX; i <= domainRightX; i++)
		{
			for (int j = domainLowerY; j <= domainUpperY; j++)
			{
				nextW[i + j * (M + 1)] = previousW[i + j * (M + 1)] - tau * r[i + j * (M + 1)];
				difference[i + j * (M + 1)] = nextW[i + j * (M + 1)] - previousW[i + j * (M + 1)];
			}
		}

		double differenceNorm = 0.0;
		double totalDifferenceNorm = 0.0;

		for (int i = domainLeftX; i <= domainRightX; i++)
		{
			for (int j = domainLowerY; j <= domainUpperY; j++)
			{
				differenceNorm += hx * hy * difference[i + j * (M + 1)] * difference[i + j * (M + 1)] * rhoX(i, M, N) * rhoY(j, M, N);
			}
		}
		MPI_Allreduce(&differenceNorm, &totalDifferenceNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		if (pow(totalDifferenceNorm, 0.5) < eps)
		{
			accuracy = 0;
		}

		double* tmp;
		tmp = nextW;
		nextW = previousW;
		previousW = tmp;
	}

	double error = 0.0;
	double totalError = 0.0;

	for (int i = domainLeftX; i <= domainRightX; i++)
	{
		for (int j = domainLowerY; j <= domainUpperY; j++)
		{
			double diff = u(A1 + i * hx, B1 + j * hy) - nextW[i + j * (M + 1)];
			error += hx * hy * diff * diff * rhoX(i, M, N) * rhoY(j, M, N);
		}
	}
	MPI_Allreduce(&error, &totalError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
	totalError = pow(totalError, 0.5);

	// end time
	wtime = MPI_Wtime() - wtime;
	MPI_Reduce(&wtime, &overall_wtime, 1, MPI_DOUBLE, MPI_MAX, masterRoot, MPI_COMM_WORLD);

	if (rank == 0)
	{
		printf("Overall time: %f\n", overall_wtime);
		printf("Iterations: %d\n", iterations);
		printf("Tau: %f\n", tau);
		printf("Error: %f\n", totalError);
	}

	MPI_Finalize(); // completion function

		delete[] nextW;
	delete[] previousW;
	delete[] r;
	delete[] Ar;
	delete[] Aw;
	delete[] B;
	delete[] difference;

	return 0;
}




# include <stdlib.h>
# include <stdio.h>
#include <iostream>
# include <math.h>
#include <string.h>
#include <mpi.h>
#include <semaphore.h>
#include <unistd.h>
using namespace std;

MPI_Status status;
int main ( int argc, char *argv[] );


int main ( int argc, char *argv[] )
/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for HEATED_PLATE_OPENMP.

  Discussion:

    This code solves the steady state heat equation on a rectangular region.

    The sequential version of this program needs approximately
    18/epsilon iterations to complete. 


    The physical region, and the boundary conditions, are suggested
    by this diagram;

                   W = 0
             +------------------+
             |                  |
    W = 100  |                  | W = 100
             |                  |
             +------------------+
                   W = 100

    The region is covered with a grid of M by N nodes, and an N by N
    array W is used to record the temperature.  The correspondence between
    array indices and locations in the region is suggested by giving the
    indices of the four corners:

                  I = 0
          [0][0]-------------[0][N-1]
             |                  |
      J = 0  |                  |  J = N-1
             |                  |
        [M-1][0]-----------[M-1][N-1]
                  I = M-1

    The steady state solution to the discrete heat equation satisfies the
    following condition at an interior grid point:

      W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    where "Central" is the index of the grid point, "North" is the index
    of its immediate neighbor to the "north", and so on.
   
    Given an approximate solution of the steady state heat equation, a
    "better" solution is given by replacing each interior point by the
    average of its 4 neighbors - in other words, by using the condition
    as an ASSIGNMENT statement:

      W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    If this process is repeated often enough, the difference between successive 
    estimates of the solution will go to zero.

    This program carries out such an iteration, using a tolerance specified by
    the user, and writes the final estimate of the solution to a file that can
    be used for graphic processing.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    18 October 2011

  Author:

    Original C version by Michael Quinn.
    This C version by John Burkardt.

  Reference:

    Michael Quinn,
    Parallel Programming in C with MPI and OpenMP,
    McGraw-Hill, 2004,
    ISBN13: 978-0071232654,
    LC: QA76.73.C15.Q55.

  Local parameters:

    Local, double DIFF, the norm of the change in the solution from one iteration
    to the next.

    Local, double MEAN, the average of the boundary values, used to initialize
    the values of the solution in the interior.

    Local, double U[M][N], the solution at the previous iteration.

    Local, double W[M][N], the solution computed at the latest iteration.
*/
{
# define M 500
# define N 500
    double my_start,my_end,my_elapsed,elapsed;
    double diff;
    double epsilon = 0.001;
    int i;
    int iterations;
    int iterations_print;
    int j;
    double mean;
    double my_diff;
    double wtime;
    double u[M][N];
    double w[M][N];

    int my_id = 0;
    int size;


    mean = 0.0;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

/*
  Set the boundary values, which don't change. 
*/
  mean = 0.0;
    for ( i = 1; i < M - 1; i++ )
    {
      w[i][0] = 100.0;
    }
    for ( i = 1; i < M - 1; i++ )
    {
      w[i][N-1] = 100.0;
    }
    for ( j = 0; j < N; j++ )
    {
      w[M-1][j] = 100.0;
    }
    for ( j = 0; j < N; j++ )
    {
      w[0][j] = 0.0;
    }
/*
  Average the boundary values, to come up with a reasonable
  initial value for the interior.
*/

    for (i = 1; i < M - 1; i++)
    {
        mean = mean + w[i][0] + w[i][N - 1];
    }
    for (j = 0; j < N; j++)
    {
        mean = mean + w[M - 1][j] + w[0][j];
    }

    mean = mean / ( double ) ( 2 * M + 2 * N - 4 );

    if(my_id == 0) {
    printf ( "\n" );
    printf ( "HEATED_PLATE_OPENMP\n" );
    printf ( "  C/OpenMP version\n" );
    printf ( "  A program to solve for the steady state temperature distribution\n" );
    printf ( "  over a rectangular plate.\n" );
    printf ( "\n" );
    printf ( "  Spatial grid of %d by %d points.\n", M, N );
    printf ( "  The iteration will be repeated until the change is <= %e\n", epsilon );
    printf ( "  Number of processes =              %d\n",size);
        printf ( "\n" );
        printf ( "  MEAN = %f\n", mean );
    }

    for(int i = max(1,my_id * M / size - 1);i < min(M - 1,(my_id + 1) * M / size + 1);i++){
        for(int j = 1;j < N - 1;j++){
            w[i][j] = mean;
        }
    }
    diff = epsilon;
    if(my_id == 0){
        iterations = 0;
        iterations_print = 1;
        printf ( "\n" );
        printf ( " Iteration  Change\n" );
        printf ( "\n" );
    }

    my_start = MPI_Wtime();
    while ( epsilon <= diff )
    {

        memcpy(&u[0][0],&w[0][0],sizeof(double) * M * N);
/*
  Determine the new estimate of the solution at the interior points.
  The new solution W is the average of north, south, east and west neighbors.
*/

        for ( i = max(1,my_id * M / size); i < min(M - 1,(my_id + 1) * M / size); i++ )
        {
            for ( j = 1; j < N - 1; j++ )
            {
            w[i][j] = ( u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] ) / 4.0;
            }
        }

/*
  C and C++ cannot compute a maximum as a reduction operation.

  Therefore, we define a private variable MY_DIFF for each thread.
  Once they have all computed their values, we use a CRITICAL section
  to update DIFF.
*/
diff = 0.0;
double buf1[N], buf2[N];
MPI_Request send_request, recv_request;
MPI_Status send_status, recv_status;
int pos1 = 0, pos2 = 0;

// Pack and send data to the right neighbor
if (my_id != size - 1) {
    MPI_Pack(&w[(my_id + 1) * M / size - 1][1], N - 2, MPI_DOUBLE, buf1, 8 * (N - 2), &pos1, MPI_COMM_WORLD);
    MPI_Isend(buf1, N - 2, MPI_DOUBLE, my_id + 1, 0, MPI_COMM_WORLD, &send_request);
}

// Pack and receive data from the left neighbor
if (my_id != 0) {
    MPI_Irecv(buf2, N - 2, MPI_DOUBLE, my_id - 1, 0, MPI_COMM_WORLD, &recv_request);
}

// Wait for the receive to complete
if (my_id != 0) {
    MPI_Wait(&recv_request, &recv_status);
    MPI_Unpack(buf2, 8 * (N - 2), &pos2, &w[my_id * M / size - 1][1], N - 2, MPI_DOUBLE, MPI_COMM_WORLD);
}

pos1 = pos2 = 0;

// Pack and send data to the left neighbor
if (my_id != 0) {
    MPI_Pack(&w[my_id * M / size][1], N - 2, MPI_DOUBLE, buf1, 8 * (N - 2), &pos1, MPI_COMM_WORLD);
    MPI_Isend(buf1, N - 2, MPI_DOUBLE, my_id - 1, 0, MPI_COMM_WORLD, &send_request);
}

// Pack and receive data from the right neighbor
if (my_id != size - 1) {
    MPI_Irecv(buf2, N - 2, MPI_DOUBLE, my_id + 1, 0, MPI_COMM_WORLD, &recv_request);
}

// Wait for the receive to complete
if (my_id != size - 1) {
    MPI_Wait(&recv_request, &recv_status);
    MPI_Unpack(buf2, 8 * (N - 2), &pos2, &w[(my_id + 1) * M / size][1], N - 2, MPI_DOUBLE, MPI_COMM_WORLD);
}

my_diff = 0.0;

for (i = max(1, my_id * M / size); i < min(M - 1, (my_id + 1) * M / size); i++) {
    for (j = 1; j < N - 1; j++) {
        if (my_diff < fabs(w[i][j] - u[i][j])) {
            my_diff = fabs(w[i][j] - u[i][j]);
        }
    }
}


        MPI_Allreduce(&my_diff,&diff,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);


        if(my_id == 0){
            iterations++;
            if ( iterations == iterations_print )
            {
            printf ( "  %8d  %f\n", iterations, diff );
            iterations_print = 2 * iterations_print;
            }
        }
    }

    my_end = MPI_Wtime();

    my_elapsed = my_end - my_start;

    MPI_Reduce(&my_elapsed,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(my_id == 0){
        printf ( "  %8d  %f\n", iterations, diff );
        printf ( "\n" );
        printf ( "  Error tolerance achieved.\n" );
        printf ( "  Wallclock time = %f\n", my_elapsed );
    }

    MPI_Finalize();
}

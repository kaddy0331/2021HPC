# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <float.h>
# include <omp.h>
# include "parallel_for.h"

int main (int argc, char * argv[]);
void ccopy ( int n, double x[], double y[] );
void cfft2 ( int n, double x[], double y[], double w[], double sgn );
void cffti ( int n, double w[] );
double cpu_time ( void );
double ggl ( double *ds );
void step ( int n, int mj, double a[], double b[], double c[], double d[], 
	double w[], double sgn );
void timestamp ( );

int num_threads;

int main (int argc, char** argv)
{
	num_threads = atoi(argv[1]);
	double ctime;
	double ctime1;
	double ctime2;
	double error;
	int first;
	double flops;
	double fnm1;
	int i;
	int icase;
	int it;
	int ln2;
	double mflops;
	int n;
	int nits = 10000;
	static double seed;
	double sgn;
	double *w;
	double *x;
	double *y;
	double *z;
	double z0;
	double z1;
	
	timestamp ( );
	printf ( "\n" );
	printf ( "FFT_SERIAL\n" );
	printf ( "  C version\n" );
	printf ( "\n" );
	printf ( "  Demonstrate an implementation of the Fast Fourier Transform\n" );
	printf ( "  of a complex data vector.\n" );
	
	printf ( "\n" );
	printf ( "  Accuracy check:\n" );
	printf ( "  num_thread:%d\n", num_threads );
	printf ( "\n" );
	printf ( "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n" );
	printf ( "\n" );
	printf ( "             N      NITS    Error         Time          Time/Call     MFLOPS\n" );
	printf ( "\n" );
	
	seed  = 331.0;
	n = 1;
	
	for ( ln2 = 1; ln2 <= 21; ln2++ )
	{
		n = 2 * n;
		
		w = ( double * ) malloc (     n * sizeof ( double ) );
		x = ( double * ) malloc ( 2 * n * sizeof ( double ) );
		y = ( double * ) malloc ( 2 * n * sizeof ( double ) );
		z = ( double * ) malloc ( 2 * n * sizeof ( double ) );
		
		first = 1;
		
		for ( icase = 0; icase < 2; icase++ )
		{
			if ( first )
			{
				for ( i = 0; i < 2 * n; i = i + 2 )
				{
					z0 = ggl ( &seed );
					z1 = ggl ( &seed );
					x[i] = z0;
					z[i] = z0;
					x[i+1] = z1;
					z[i+1] = z1;
				}
			} 
			else
			{
				for ( i = 0; i < 2 * n; i = i + 2 )
				{
					z0 = 0.0;              /* real part of array */
					z1 = 0.0;              /* imaginary part of array */
					x[i] = z0;
					z[i] = z0;           /* copy of initial real data */
					x[i+1] = z1;
					z[i+1] = z1;         /* copy of initial imag. data */
				}
			}
			
			cffti ( n, w );
			
			if ( first )
			{
				sgn = + 1.0;
				cfft2 ( n, x, y, w, sgn );
				sgn = - 1.0;
				cfft2 ( n, y, x, w, sgn );
				
				fnm1 = 1.0 / ( double ) n;
				error = 0.0;
				for ( i = 0; i < 2 * n; i = i + 2 )
				{
					error = error 
					+ pow ( z[i]   - fnm1 * x[i], 2 )
					+ pow ( z[i+1] - fnm1 * x[i+1], 2 );
				}
				error = sqrt ( fnm1 * error );
				printf ( "  %12d  %8d  %12e", n, nits, error );
				first = 0;
			}
			else
			{
				ctime1 = cpu_time ( );
				for ( it = 0; it < nits; it++ )
				{
					sgn = + 1.0;
					cfft2 ( n, x, y, w, sgn );
					sgn = - 1.0;
					cfft2 ( n, y, x, w, sgn );
				}
				ctime2 = cpu_time ( );
				ctime = ctime2 - ctime1;
				
				flops = 2.0 * ( double ) nits * ( 5.0 * ( double ) n * ( double ) ln2 );
				
				mflops = flops / 1.0E+06 / ctime;
				
				printf ( "  %12e  %12e  %12f\n", ctime, ctime / ( double ) ( 2 * nits ), mflops );
			}
		}
		if ( ( ln2 % 4 ) == 0 ) 
		{
			nits = nits / 10;
		}
		if ( nits < 1 ) 
		{
			nits = 1;
		}
		free ( w );
		free ( x );
		free ( y );
		free ( z );
	}
	printf ( "\n" );
	printf ( "FFT_SERIAL:\n" );
	printf ( "  Normal end of execution.\n" );
	printf ( "\n" );
	timestamp ( );
	
	return 0;
}

void ccopy ( int n, double x[], double y[] )


{
	int i;
	
	for ( i = 0; i < n; i++ )
	{
		y[i*2+0] = x[i*2+0];
		y[i*2+1] = x[i*2+1];
	}
	return;
}

/*
  struct ccopy_args {
  int n;
  double *x;
  double *y;
  };
  
  void *ccopy_worker(void *args);
  
  void ccopy(int n, double x[], double y[]) {
  struct ccopy_args ccopy_args = {n, x, y};  // 具名结构体对象
  // 使用 parallel_for 并行化循环
  parallel_for(0, n, 1, (void *(*)(void *, int))ccopy_worker, (void *)&ccopy_args, 2);
  }
  
  void *ccopy_worker(void *args) {
  struct ccopy_args *ccopy_args = (struct ccopy_args *)args;
  int n = ccopy_args->n;
  double *x = ccopy_args->x;
  double *y = ccopy_args->y;
  
  // 并行复制
  for (int i = 0; i < n; i++) {
  y[i * 2 + 0] = x[i * 2 + 0];
  y[i * 2 + 1] = x[i * 2 + 1];
  }
  
  return NULL;
  }
 */


void cfft2 ( int n, double x[], double y[], double w[], double sgn )


{
	int j;
	int m;
	int mj;
	int tgle;
	
	m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) );
	mj   = 1;
	
	tgle = 1;
	step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
	
	if ( n == 2 )
	{
		return;
	}
	
	for ( j = 0; j < m - 2; j++ )
	{
		mj = mj * 2;
		if ( tgle )
		{
			step ( n, mj, &y[0*2+0], &y[(n/2)*2+0], &x[0*2+0], &x[mj*2+0], w, sgn );
			tgle = 0;
		}
		else
		{
			step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
			tgle = 1;
		}
	}
	
	if ( tgle ) 
	{
		ccopy ( n, y, x );
	}
	
	mj = n / 2;
	step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
	
	return;
}
/*
  void cffti ( int n, double w[] )
  
  
  {
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;
  
  n2 = n / 2;
  aw = 2.0 * pi / ( ( double ) n );
  
  for ( i = 0; i < n2; i++ )
  {
  arg = aw * ( ( double ) i );
  w[i*2+0] = cos ( arg );
  w[i*2+1] = sin ( arg );
  }
  return;
  }
 */


struct CFFTI_Arg {
	int n;
	double *w;
};

void* cffti_functor(void* args, int i) {
	CFFTI_Arg* arg = static_cast<CFFTI_Arg*>(args);
	
	if (i < arg->n / 2) {
		double aw = 2.0 * M_PI / static_cast<double>(arg->n);
		double angle = aw * static_cast<double>(i);
		
		arg->w[i * 2 + 0] = cos(angle);
		arg->w[i * 2 + 1] = sin(angle);
	}
	
	return nullptr;
}

void cffti(int n, double w[]) {
	int n2 = n / 2;
	
	CFFTI_Arg arg;
	arg.n = n;
	arg.w = w;
	
	parallel_for(0, n2, 1, cffti_functor, &arg, num_threads); 
}






double cpu_time ( void )

{
	double value;
	
	value = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC;
	
	return value;
}

double ggl ( double *seed )


{
	double d2 = 0.2147483647e10;
	double t;
	double value;
	
	t = *seed;
	t = fmod ( 16807.0 * t, d2 );
	*seed = t;
	value = ( t - 1.0 ) / ( d2 - 1.0 );
	
	return value;
}

void step ( int n, int mj, double a[], double b[], double c[],
	double d[], double w[], double sgn )

{
	double ambr;
	double ambu;
	int j;
	int ja;
	int jb;
	int jc;
	int jd;
	int jw;
	int k;
	int lj;
	int mj2;
	double wjw[2];
	
	mj2 = 2 * mj;
	lj  = n / mj2;
	
	for ( j = 0; j < lj; j++ )
	{
		jw = j * mj;
		ja  = jw;
		jb  = ja;
		jc  = j * mj2;
		jd  = jc;
		
		wjw[0] = w[jw*2+0]; 
		wjw[1] = w[jw*2+1];
		
		if ( sgn < 0.0 ) 
		{
			wjw[1] = - wjw[1];
		}
		
		for ( k = 0; k < mj; k++ )
		{
			c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];
			c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];
			
			ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0];
			ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1];
			
			d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;
			d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;
		}
	}
	return;
}

/*
  struct step_args {
  int n;
  int mj;
  double *a;
  double *b;
  double *c;
  double *d;
  double *w;
  double sgn;
  int lj;
  };
  
  void *step_parallel(void *args) {
  struct step_args *sargs = (struct step_args *)args;
  int n = sargs->n;
  int mj = sargs->mj;
  double *a = sargs->a;
  double *b = sargs->b;
  double *c = sargs->c;
  double *d = sargs->d;
  double *w = sargs->w;
  double sgn = sargs->sgn;
  
  int mj2 = 2 * mj;
  int lj = n /mj2;
  double wjw[2];
  
  for (int j = 0; j <lj; j++) {
  int jw = j* mj;
  
  wjw[0] = w[jw * 2 +0];
  wjw[1] = (sgn < 0.0) ? -w[jw * 2 + 1] : w[jw * 2 + 1];
  
  for (int k = 0; k < mj ;k++) {
  int ja =  jw + k;
  int jb = ja;
  int jc = j * mj2 + k;
  int jd = jc;
  
  c[(jc)*2 + 0] = a[(ja)*2 + 0] + b[(jb)*2 + 0];
  c[(jc)*2 + 1] = a[(ja)*2 + 1] + b[(jb)*2 + 1];
  
  double ambr = a[(ja)*2 + 0] - b[(jb)*2 + 0];
  double ambu = a[(ja)*2 + 1] - b[(jb)*2 + 1];
  
  d[(jd)*2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
  d[(jd)*2 + 1] = wjw[1] * ambr - wjw[0] * ambu;
  }
  }
  
  return NULL;
  }
  
  void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn) {
  int mj2 = 2 * mj;
  int lj = n / mj2;
  
  struct step_args step_args = {n, mj, a, b, c, d, w, sgn, lj};
  
  parallel_for(0, lj, 1, (void *(*)(void *, int))step_parallel, (void *)&step_args, 4);
  }
 */
void timestamp ( )


{
# define TIME_SIZE 40
	
	static char time_buffer[TIME_SIZE];
	const struct tm *tm;
	time_t now;
	
	now = time ( NULL );
	tm = localtime ( &now );
	
	strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );
	
	printf ( "%s\n", time_buffer );
	
	return;
# undef TIME_SIZE
}


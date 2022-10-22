/*
 * small_mc.cu
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

// Absorption Coefficient in 1/cm
#define MU_A 5

// Scattering Coefficient in 1/cm
#define MU_S 95

// Scattering Anisotropy -1 <= g <= 1
#define G 0.5

#define BINS 101

typedef struct {
    double xPosition;
    double yPosition;
    double zPosition;
    double deltaXPosition;
    double deltaYPosition;
    double deltaZPosition;
    double weight;
} Photon;

// TODO: Doxygen
void launch();

// TODO: Doxygen
void bounce();

// TODO: Doxygen
void move();

// TODO: Doxygen
void absorb();

// TODO: Doxygen
void scatter();

// TODO: Doxygen
void print_results();

int main(void) {
    double n = 1.5;				/* Index of refraction of medium */
    double microns_per_bin = 20;/* Thickness of one bin layer */
    long   i, photons = 100000;
    double rs, rd, bit, albedo, crit_angle, bins_per_mfp, heat[BINS];

	albedo = MU_S / (MU_S + MU_A);
	rs = (n-1.0)*(n-1.0)/(n+1.0)/(n+1.0);	/* specular reflection */
	crit_angle = sqrt(1.0-1.0/n/n);			/* cos of critical angle */
	bins_per_mfp = 1e4 / microns_per_bin / (MU_A + MU_S);
	
	for (i = 1; i <= photons; i++){
		launch ();
		while (weight > 0) {
			move ();
			absorb ();
			scatter ();
		}
	}	

	print_results();

	return 0;
}

// TODO: Doxygen
void launch() /* Start the photon */
{
	x = 0.0; y = 0.0; z = 0.0;		  
	u = 0.0; v = 0.0; w = 1.0;		
	weight = 1.0 - rs;
}

// TODO: Doxygen
void bounce () /* Interact with top surface */
{
double t, temp, temp1,rf;
	w = -w;
	z = -z;
	if (w <= crit_angle) return;  			/* total internal reflection */	

	t       = sqrt(1.0-n*n*(1.0-w*w));    	/* cos of exit angle */
	temp1   = (w - n*t)/(w + n*t);
	temp    = (t - n*w)/(t + n*w);
	rf      = (temp1*temp1+temp*temp)/2.0;	/* Fresnel reflection */
	rd     += (1.0-rf) * weight;
	weight -= (1.0-rf) * weight;
}

// TODO: Doxygen
void move() /* move to next scattering or absorption event */
{
double d = -log((rand()+1.0)/(RAND_MAX+1.0));
	x += d * u;
	y += d * v;
	z += d * w;  
	if ( z<=0 ) bounce();
}

// TODO: Doxygen
void absorb () /* Absorb light in the medium */
{
int bin=z*bins_per_mfp;

	if (bin >= BINS) bin = BINS-1;	
	heat[bin] += (1.0-albedo)*weight;
	weight *= albedo;
	if (weight < 0.001){ /* Roulette */
		bit -= weight;
		if (rand() > 0.1*RAND_MAX) weight = 0; else weight /= 0.1;
		bit += weight;
	}
}

// TODO: Doxygen
void scatter() /* Scatter photon and establish new direction */
{
double x1, x2, x3, t, mu;

	for(;;) {								/*new direction*/
		x1=2.0*rand()/RAND_MAX - 1.0; 
		x2=2.0*rand()/RAND_MAX - 1.0; 
		if ((x3=x1*x1+x2*x2)<=1) break;
	}	
	if (G==0) {  /* isotropic */
		u = 2.0 * x3 -1.0;
		v = x1 * sqrt((1-u*u)/x3);
		w = x2 * sqrt((1-u*u)/x3);
		return;
	} 

	mu = (1-G*G)/(1-G+2.0*G*rand()/RAND_MAX);
	mu = (1 + G*G-mu*mu)/2.0/G;
	if ( fabs(w) < 0.9 ) {	
		t = mu * u + sqrt((1-mu*mu)/(1-w*w)/x3) * (x1*u*w-x2*v);
		v = mu * v + sqrt((1-mu*mu)/(1-w*w)/x3) * (x1*v*w+x2*u);
		w = mu * w - sqrt((1-mu*mu)*(1-w*w)/x3) * x1;
	} else {
		t = mu * u + sqrt((1-mu*mu)/(1-v*v)/x3) * (x1*u*v + x2*w);
		w = mu * w + sqrt((1-mu*mu)/(1-v*v)/x3) * (x1*v*w - x2*u);
		v = mu * v - sqrt((1-mu*mu)*(1-v*v)/x3) * x1;
	}
	u = t;
}

/**
 * 
 */
void print_results() {
    printf("Small Monte Carlo by Scott Prahl (https://omlc.org)\n");
    printf("1 W/cm^2 Uniform Illumination of Semi-Infinite Medium\n\n");

    printf("Scattering = %8.3f/cm\n", MU_S);
    printf("Absorption = %8.3f/cm\n", MU_A);
    printf("Anisotropy = %8.3f\n", G);
    printf("Refraction Index = %8.3f\n", n);
    printf("Number of Photons = %8ld\n\n", photons);

    printf("Specular Reflection = %10.5f\n", rs);
    printf("Backscattered Reflection = %10.5f\n\n", rd / (bit + photons));

	printf("Depth         Heat\n[microns]     [W/cm^3]\n");

	for (int i = 0; i < BINS - 1; i++) {
		printf("%6.0f    %12.5f\n", 
            i * microns_per_bin, 
            heat[i] / microns_per_bin * 1e4 / (bit + photons));
	}

	printf("Extra Heat [W/cm^3]  %12.5f\n", heat[BINS - 1] / (bit + photons));
}
#pragma once
struct Calcuation_Data {
	unsigned char *screenBufferCPU;
	double *escapeBufferCPU;
	bool *updatePixel;

	bool screenStill;
	bool screenOptimization;
	int prevMaxIter;
};

// Exploit locality by referring to fixed memory location
struct Calculation_Thread {
	// Coordinate
	double ci[8];
	double cr[8];

	// Iteration Calculations
	double cici[8];
	double crcr[8];
	double z[8];

};
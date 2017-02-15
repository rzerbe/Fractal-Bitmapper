#pragma once
struct Calculation_Data {
	unsigned char *screenBufferCPU;
	double *escapeBufferCPU;
	double *escapeBufferSuperSampling;
	double *magnitude;
	bool *updatePixel;

	bool screenStill;
	bool screenOptimization;
	bool periodOptimization;
	int prevMaxIter;
};
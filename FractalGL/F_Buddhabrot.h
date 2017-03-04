#pragma once
#include "Calculation.h"
#include "Configuration.h"
class F_Buddhabrot
{
public:
	F_Buddhabrot(Calculation_Data &calculation, Configuration &config);
	~F_Buddhabrot();
	void fractal_render(int output);

	Calculation_Data &calculation;
	Configuration &config;
	unsigned int iteration_min_red;
	unsigned int iteration_min_green;
	unsigned int iteration_min_blue;

	unsigned int iteration_max_red;
	unsigned int iteration_max_green;
	unsigned int iteration_max_blue;

	unsigned int sample_size_red;
	unsigned int sample_size_green;
	unsigned int sample_size_blue;

	unsigned int highest_escape_red;
	unsigned int highest_escape_green;
	unsigned int highest_escape_blue;

	unsigned int *iteration_red;
	unsigned int *iteration_green;
	unsigned int *iteration_blue;

private:
	// Generators
	void c_fractal_64(int color, int n_init, int n_fin, int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads);
	void avx_fractal_32(int color, int n_init, int n_fin, int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads);

	// Modifiers
	void filter_histogram();

	// Formatters
	void write_region(int x_init, int y_init, int x_fin, int y_fin);
	void write_buffer();
};
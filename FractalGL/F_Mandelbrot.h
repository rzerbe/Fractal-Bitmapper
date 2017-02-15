#pragma once
#include "Calculation.h"
#include "Configuration.h"
class F_Mandelbrot
{
public:
	F_Mandelbrot(Calculation_Data &calculation, Configuration &config);
	~F_Mandelbrot();

	void fractal_render(int output);
	Calculation_Data &calculation;
	Configuration &config;

private:
	// Generators
	void avx_fractal_32(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads);
	void avx_fractal_64(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads);
	void c_fractal_64(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads);

	// Modifiers
	void downsampler();
	void downsample_region(int x_init, int y_init, int x_fin, int y_fin);
	void normalize();

	// Formatters
	void write_pixel(int x, int y);
	void write_region(int x_init, int y_init, int x_fin, int y_fin);
	void write_buffer();
};

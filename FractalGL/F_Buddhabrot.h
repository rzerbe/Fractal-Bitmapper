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
	uint64_t iteration_min_red;
	uint64_t iteration_min_green;
	uint64_t iteration_min_blue;

	uint64_t iteration_max_red;
	uint64_t iteration_max_green;
	uint64_t iteration_max_blue;

	uint64_t sample_size_red;
	uint64_t sample_size_green;
	uint64_t sample_size_blue;

	uint64_t highest_escape_red;
	uint64_t highest_escape_green;
	uint64_t highest_escape_blue;

	uint32_t *iteration_red;
	uint32_t *iteration_green;
	uint32_t *iteration_blue;

	/* The state must be seeded so that it is not everywhere zero. */
	uint64_t *s;

private:
	// Generators
	void c_fractal_64(int color, uint64_t n_max, int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads);
	void avx_fractal_32(int color, uint64_t n_max, int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads);
	inline uint64_t xorshift128plus(int thread);

	// Modifiers
	void filter_histogram();

	// Formatters
	void write_region(int x_init, int y_init, int x_fin, int y_fin);
	void write_buffer();
};
// Includes
#include <cstdio>
#include <fstream>
#include <iomanip>  
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// CPU
#include <chrono>
#include <thread>
#include <immintrin.h>
#include <intrin.h>

#include "F_Buddhabrot.h"
#include "Calculation.h"
#include "Configuration.h"

// Mandelbrot


#include <stdint.h>
//https://en.wikipedia.org/wiki/Xorshift

inline uint64_t F_Buddhabrot::xorshift128plus(int thread) {
	uint64_t x = s[0+2*thread];
	uint64_t const y = s[1+2*thread];
	s[0+2*thread] = y;
	x ^= x << 23; // a
	s[1+2*thread] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
	return s[1+2*thread] + y;
}

F_Buddhabrot::F_Buddhabrot(Calculation_Data &calc, Configuration &conf)
	: calculation(calc), config(conf)
{

	s = new uint64_t[config.threadMax * 2];

	for (int i = 0; i < config.threadMax * 2; i++)
	{
		s[i] = i;
	}

	iteration_max_red = config.iteration_max_red;
	iteration_max_green = config.iteration_max_green;
	iteration_max_blue = config.iteration_max_blue;

	iteration_min_red = config.iteration_min_red;
	iteration_min_green = config.iteration_min_green;
	iteration_min_blue = config.iteration_min_blue;

	sample_size_red = config.sample_size_red;
	sample_size_green = config.sample_size_green;
	sample_size_blue = config.sample_size_blue;

	highest_escape_red = 0;
	highest_escape_green = 0;
	highest_escape_blue = 0;

	iteration_red = new uint32_t[config.resolutionX * config.resolutionY];
	iteration_green = new uint32_t[config.resolutionX * config.resolutionY];
	iteration_blue = new uint32_t[config.resolutionX * config.resolutionY];
}

F_Buddhabrot::~F_Buddhabrot()
{
	delete[] iteration_red;
	delete[] iteration_green;
	delete[] iteration_blue;
	delete[] s;
	printf("F_Buddhabrot Deleted\n");
}

void F_Buddhabrot::fractal_render(int output)
{
	for (int j = 0; j < config.resolutionY; j++)
	{
		for (int i = 0; i < config.resolutionX; i++)
		{
			iteration_red[j*config.resolutionX + i] = 0;
			iteration_green[j*config.resolutionX + i] = 0;
			iteration_blue[j*config.resolutionX + i] = 0;
		}
	}
	auto begin = std::chrono::high_resolution_clock::now();
	
	for (int color = 0; color < 3; ++color)
	{
		uint64_t sample_size = 0;
		if (color == 0)
		{
			sample_size = sample_size_red;
		}
		else if (color == 1)
		{
			sample_size = sample_size_green;
		}
		else if (color == 2)
		{
			sample_size = sample_size_blue;
		}

		std::vector<std::thread> fractalT(config.threadMax);
		for (int thread = 0; thread < config.threadMax; ++thread)
		{
			//fractalT[thread] = std::thread(&F_Buddhabrot::c_fractal_64, this, color, sample_size / config.threadMax, 0, 0, config.resolutionX, config.resolutionY, thread, 0);
			fractalT[thread] = std::thread(&F_Buddhabrot::avx_fractal_32, this, color, sample_size / config.threadMax, 0, 0, config.resolutionX, config.resolutionY, thread, 0);
			printf("Samples [%jd, %jd] on Thread %d\n", sample_size / config.threadMax * thread, sample_size / config.threadMax * (thread + 1), color);
		}

		for (int thread = 0; thread < config.threadMax; ++thread)
		{
			fractalT[thread].join();
		}
	}

	// Post Processing
	write_buffer();

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("Fractal Generate Time %d\n",
		elasped
	);
}

void F_Buddhabrot::c_fractal_64(int color, uint64_t n_max, int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads)
{
	auto begin = std::chrono::high_resolution_clock::now();

	uint64_t iteration_max = 0;
	uint64_t iteration_min = 0;
	uint32_t *data = NULL;

	if (color == 0)
	{
		iteration_max = iteration_max_red;
		iteration_min = iteration_min_red;
		data = iteration_red;
	}
	else if (color == 1)
	{
		iteration_max = iteration_max_green;
		iteration_min = iteration_min_green;
		data = iteration_green;
	}
	else if (color == 2)
	{
		iteration_max = iteration_max_blue;
		iteration_min = iteration_min_blue;
		data = iteration_blue;
	}
	else
	{
		// grayscale
	}

	double scalecr = (double)(config.xmax - config.xmin) / (UINTMAX_MAX);
	double scaleci = (double)(config.ymax - config.ymin) / (UINTMAX_MAX);
	double xWindow = config.xmax - config.xmin;
	double yWindow = config.ymax - config.ymin;
	uint64_t k;
	double ci, cr, cici, q;
	double zr, zrzr;
	double zi, zizi;
	double zr_prev;

	for (uint64_t n = 0; n < n_max; ++n)
	{
		cr = config.xmin + xorshift128plus(threadID) * scalecr;

		ci = config.ymin + xorshift128plus(threadID) * scaleci;

		// Bulb checking
		cici = ci * ci;
		q = (cr - 0.25)*(cr - 0.25) + cici;
		if (q*(q + cr - 0.25) - 0.25*cici < 0)
		{
			continue;
		}
		else if ((cr + 1)*(cr + 1) + cici - 0.0625 < 0)
		{
			continue;
		}

		zr = 0;
		zi = 0;

		for (k = 1; k < iteration_max; ++k)
		{
			zr_prev = zr;
			zrzr = zr*zr;
			zizi = zi*zi;
			zr = zrzr - zizi + cr;
			zi = 2 * zr_prev * zi + ci;

			if (zrzr + zizi > 4.0)
			{
				break;
			}
		}
		/*// Anti-Buddhabrot!
		if (t != iteration_max)
		{
			continue;
		}

		zr = 0;
		zi = 0;
		*/

		if (k < iteration_min || k == iteration_max)
		{
			continue;  // Reject Random Point
		}

		zr = 0;
		zi = 0;
		

		for (k = 1; k < iteration_max; ++k)
		{
			zr_prev = zr;
			zrzr = zr*zr;
			zizi = zi*zi;
			zr = zrzr - zizi + cr;
			zi = 2 * zr_prev * zi + ci;

			if (zrzr + zizi > 4.0)
			{
				break;
			}
			int jn = (int)((config.resolutionY / yWindow)*(zi - config.ymin));
			int in = (int)((config.resolutionX / xWindow)*(zr - config.xmin));
			if (in < config.resolutionX && jn < config.resolutionY && in >= 0 && jn >= 0)
			{
				++data[jn*config.resolutionX + in];
			}
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("Thread %d Generate Time %d\n",
		threadID, elasped
	);
}

void F_Buddhabrot::avx_fractal_32(int color, uint64_t n_max, int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads)
{
	auto begin = std::chrono::high_resolution_clock::now();
	uint64_t iteration_max = 0;
	uint64_t iteration_min = 0;
	uint32_t *data = NULL;

	if (color == 0)
	{
		iteration_max = iteration_max_red;
		iteration_min = iteration_min_red;
		data = iteration_red;
	}
	else if (color == 1)
	{
		iteration_max = iteration_max_green;
		iteration_min = iteration_min_green;
		data = iteration_green;
	}
	else if (color == 2)
	{
		iteration_max = iteration_max_blue;
		iteration_min = iteration_min_blue;
		data = iteration_blue;
	}
	else
	{
		// Grayscale
	}

	__m256 ymm_scalecr = _mm256_set1_ps((float)(config.xmax - config.xmin) / (UINT32_MAX));
	__m256 ymm_scaleci = _mm256_set1_ps((float)(config.ymax - config.ymin) / (UINT32_MAX));
	__m256 ymm_dx = _mm256_set1_ps((float)((config.xmax - config.xmin) / config.resolutionX));
	__m256 ymm_dy = _mm256_set1_ps((float)((config.ymax - config.ymin) / config.resolutionY));
	__m256 ymm_invdx = _mm256_set1_ps((float)(config.resolutionX / (config.xmax - config.xmin)));
	__m256 ymm_invdy = _mm256_set1_ps((float)(config.resolutionY / (config.ymax - config.ymin)));

	__m256 ymm_wx = _mm256_set1_ps((float)config.xmin);
	__m256 ymm_wy = _mm256_set1_ps((float)config.ymin);
	__m256 ymm_wResX = _mm256_set1_ps((float)config.resolutionX);
	__m256 ymm_wResY = _mm256_set1_ps((float)config.resolutionY);

	__m256 ymm_one = _mm256_set1_ps(1.0f);
	__m256 ymm_four = _mm256_set1_ps(4.0f);
	__m256 ymm_maxIter = _mm256_set1_ps((float)(iteration_max));
	uint32_t temp[8];

	for (uint64_t n = 0; n < n_max; n += 8)
	{

		// Choose random points
		for (int i = 0; i < 8; i += 2)
		{
			uint64_t rand = xorshift128plus(threadID);
			temp[i] = (uint32_t)(rand);
			temp[i + 1] = (uint32_t)(rand >> 32);
		}
		__m256 ymm_cr_rand = _mm256_set_ps((float)temp[0], (float)temp[1], (float)temp[2], (float)temp[3], (float)temp[4], (float)temp[5], (float)temp[6], (float)temp[7]);

		for (int i = 0; i < 8; i += 2)
		{
			uint64_t rand = xorshift128plus(threadID);
			temp[i] = (uint32_t)(rand);
			temp[i + 1] = (uint32_t)(rand >> 32);
		}
		__m256 ymm_ci_rand = _mm256_set_ps((float)temp[0], (float)temp[1], (float)temp[2], (float)temp[3], (float)temp[4], (float)temp[5], (float)temp[6], (float)temp[7]);

		__m256 ymm_cr = _mm256_mul_ps(ymm_cr_rand, ymm_scalecr);		// cr[] = (i) * (scalecr)
		ymm_cr = _mm256_add_ps(ymm_cr, ymm_wx);							// cr[] = (i*scaleci) + (x)
		__m256 ymm_ci = _mm256_mul_ps(ymm_ci_rand, ymm_scaleci);		// ci[] = (j) * (scaleci)
		ymm_ci = _mm256_add_ps(ymm_ci, ymm_wy);							// ci[] = (j*scaleci) + (y)

		// Bulb checking
		
		// q = (cr - 0.25)*(cr - 0.25) + ci*ci;
		__m256 ymm_cici = _mm256_mul_ps(ymm_ci, ymm_ci);
		__m256 ymm_factorcr = _mm256_sub_ps(ymm_cr, _mm256_set1_ps(0.25f));
		__m256 ymm_q = _mm256_mul_ps(ymm_factorcr, ymm_factorcr);
		ymm_q = _mm256_add_ps(ymm_q, ymm_cici);

		// if (q*(q + cr - 0.25) - (0.25*cici)) > 0	// point out first bulb
		__m256 ymm_term = _mm256_add_ps(ymm_q, ymm_factorcr);
		ymm_term = _mm256_mul_ps(ymm_q, ymm_term);
		ymm_term = _mm256_sub_ps(ymm_term, _mm256_mul_ps(_mm256_set1_ps(0.25f), ymm_cici));

		__m256 ymm_check = _mm256_cmp_ps(ymm_term, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
		ymm_check = _mm256_and_ps(ymm_check, ymm_one);
		ymm_cr = _mm256_div_ps(ymm_cr, ymm_check);
		ymm_ci = _mm256_div_ps(ymm_ci, ymm_check);

		// else if ((cr + 1)*(cr + 1) + cici - 0.0625 > 0) // point outside second bulb
		ymm_term = _mm256_add_ps(ymm_cr, ymm_one);
		ymm_term = _mm256_mul_ps(ymm_term, ymm_term);
		ymm_term = _mm256_add_ps(ymm_term, ymm_cici);
		ymm_term = _mm256_sub_ps(ymm_term, _mm256_set1_ps(0.0625f));

		ymm_check = _mm256_cmp_ps(ymm_term, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
		ymm_check = _mm256_and_ps(ymm_check, ymm_one);
		ymm_cr = _mm256_div_ps(ymm_cr, ymm_check);
		ymm_ci = _mm256_div_ps(ymm_ci, ymm_check);
		

		// Test Points

		uint64_t iter = 0;	// Counter for highest iteration
		int test = 0; // bits [7:0] used in comparison


		__m256 ymm_iter = _mm256_xor_ps(ymm_dx, ymm_dx);				// iter[] = 0
		__m256 ymm_zr = ymm_iter;										// xi[] = 0
		__m256 ymm_zi = ymm_iter;										// yi[] = 0

		do
		{
			__m256 ymm_zrzr = _mm256_mul_ps(ymm_zr, ymm_zr);			// zrzr[] = (zr[]) * (zr[])
			__m256 ymm_zizi = _mm256_mul_ps(ymm_zi, ymm_zi);			// zizi[] = (zi[]) * (zi[])
			__m256 ymm_temp = _mm256_add_ps(ymm_zrzr, ymm_zizi);		// temp[] = (zr[]*zr[]) + (zi[]*zi[])

																		// zr*zr+zi*zi < 4
			ymm_temp = _mm256_cmp_ps(ymm_temp, ymm_four, _CMP_LT_OQ);	// temp[] = {0^32 | 1^32}
			test = _mm256_movemask_ps(ymm_temp) & 255;					// test =	(0^8  | 1^8) AND 11111111
			ymm_temp = _mm256_and_ps(ymm_temp, ymm_one);				// temp[] = {0.0f | 1.0f}
			ymm_iter = _mm256_add_ps(ymm_iter, ymm_temp);				// iter[] = (iter[]) + (temp[])

			ymm_temp = _mm256_mul_ps(ymm_zr, ymm_zi);					// temp[] = (zr[]) * (zi[])
			ymm_zr = _mm256_sub_ps(ymm_zrzr, ymm_zizi);					// zr[] = (zr[]*zr[]) - (zi[]*zi[])
			ymm_zr = _mm256_add_ps(ymm_zr, ymm_cr);						// zr[] = (zr[]*zr[]-zi[]*zi[]) + (cr)
			ymm_zi = _mm256_add_ps(ymm_temp, ymm_temp);					// zi[] = (xi*yi) + (xi*yi)
			ymm_zi = _mm256_add_ps(ymm_zi, ymm_ci);						// zi[] = (2*xi*yi) + (ci)	

			++iter;
		} while ((test != 0) && (iter < iteration_max));

		ymm_check = _mm256_cmp_ps(ymm_iter, ymm_maxIter, _CMP_LT_OQ);
		ymm_check = _mm256_and_ps(ymm_check, ymm_one);	// 0 = reject
		ymm_cr = _mm256_div_ps(ymm_cr, ymm_check);
		ymm_ci = _mm256_div_ps(ymm_ci, ymm_check);
		
		// Perform
		iter = 0;		// Counter for highest iteration
		test = 0;		// bits [7:0] used in comparison

		ymm_iter = _mm256_xor_ps(ymm_dx, ymm_dx);						// iter[] = 0
		ymm_zr = ymm_iter;												// xi[] = 0
		ymm_zi = ymm_iter;												// yi[] = 0
		
		do
		{
			__m256 ymm_zrzr = _mm256_mul_ps(ymm_zr, ymm_zr);			// zrzr[] = (zr[]) * (zr[])
			__m256 ymm_zizi = _mm256_mul_ps(ymm_zi, ymm_zi);			// zizi[] = (zi[]) * (zi[])
			__m256 ymm_temp = _mm256_add_ps(ymm_zrzr, ymm_zizi);		// temp[] = (zr[]*zr[]) + (zi[]*zi[])

																		// zr*zr+zi*zi < 4
			ymm_temp = _mm256_cmp_ps(ymm_temp, ymm_four, _CMP_LT_OQ);	// temp[] = {0^32 | 1^32}
			test = _mm256_movemask_ps(ymm_temp) & 255;					// test =	(0^8  | 1^8) AND 11111111

			ymm_temp = _mm256_mul_ps(ymm_zr, ymm_zi);					// temp[] = (zr[]) * (zi[])
			ymm_zr = _mm256_sub_ps(ymm_zrzr, ymm_zizi);					// zr[] = (zr[]*zr[]) - (zi[]*zi[])
			ymm_zr = _mm256_add_ps(ymm_zr, ymm_cr);						// zr[] = (zr[]*zr[]-zi[]*zi[]) + (cr)
			ymm_zi = _mm256_add_ps(ymm_temp, ymm_temp);					// zi[] = (xi*yi) + (xi*yi)
			ymm_zi = _mm256_add_ps(ymm_zi, ymm_ci);						// zi[] = (2*xi*yi) + (ci)	

			// Convert to pixel(x,y)
			__m256 ymm_jn = _mm256_sub_ps(ymm_zi, ymm_wy);
			ymm_jn = _mm256_mul_ps(ymm_invdy, ymm_jn);
			__m256 ymm_in = _mm256_sub_ps(ymm_zr, ymm_wx);
			ymm_in = _mm256_mul_ps(ymm_invdx, ymm_in);

		/*	// Reject on in > config.resolutionX
			__m256 ymm_checkPixel = _mm256_cmp_ps(ymm_in, ymm_wResX, _CMP_LT_OQ);
			ymm_checkPixel = _mm256_and_ps(ymm_checkPixel, ymm_one);
			ymm_jn = _mm256_div_ps(ymm_jn, ymm_checkPixel);
			ymm_in = _mm256_div_ps(ymm_in, ymm_checkPixel);

			// Reject on jn > config.resolutionY
			ymm_checkPixel = _mm256_cmp_ps(ymm_jn, ymm_wResY, _CMP_LT_OQ);
			ymm_checkPixel = _mm256_and_ps(ymm_checkPixel, ymm_one);
			ymm_jn = _mm256_div_ps(ymm_jn, ymm_checkPixel);
			ymm_in = _mm256_div_ps(ymm_in, ymm_checkPixel);

			// Reject on in < 0
			ymm_checkPixel = _mm256_cmp_ps(ymm_in, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
			ymm_checkPixel = _mm256_and_ps(ymm_checkPixel, ymm_one);
			ymm_jn = _mm256_div_ps(ymm_jn, ymm_checkPixel);
			ymm_in = _mm256_div_ps(ymm_in, ymm_checkPixel);

			// Reject on jn < 0
			ymm_checkPixel = _mm256_cmp_ps(ymm_jn, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
			ymm_checkPixel = _mm256_and_ps(ymm_checkPixel, ymm_one);
			ymm_jn = _mm256_div_ps(ymm_jn, ymm_checkPixel);
			ymm_in = _mm256_div_ps(ymm_in, ymm_checkPixel);   */
			__m256i ymm_jn_int = _mm256_cvtps_epi32(ymm_jn);
			__m256i ymm_in_int = _mm256_cvtps_epi32(ymm_in);
			for (int k = 0; k < 8; ++k)
			{
				int in = ymm_in_int.m256i_i32[k];
				int jn = ymm_jn_int.m256i_i32[k];
				if (in >= 0 && in < config.resolutionX)
				{
					if (jn >= 0 && jn < config.resolutionY)
					{
						++data[jn*config.resolutionX + in];
					}
				}

			}

			++iter;
		} while ((test != 0) && (iter < iteration_max));

	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("AVX_32 Thread #%d finished in %d milliseconds. \n",
		threadID, elasped
	);
}

void F_Buddhabrot::write_buffer()
{
	auto begin = std::chrono::high_resolution_clock::now();

	uint64_t rx = 0, ry = 0, gx = 0, gy = 0, bx = 0, by = 0;

	for (int y = 0; y < config.resolutionY; ++y)
	{
		for (int x = 0; x < config.resolutionX; ++x)
		{
			if (iteration_red[(y*config.resolutionX + x)] > highest_escape_red)
			{
				highest_escape_red = iteration_red[(y*config.resolutionX + x)];
				rx = x;
				ry = y;
			}
			if (iteration_green[(y*config.resolutionX + x)] > highest_escape_green)
			{
				highest_escape_green = iteration_green[(y*config.resolutionX + x)];
				gx = x;
				gy = y;
			}
			if (iteration_blue[(y*config.resolutionX + x)] > highest_escape_blue)
			{
				highest_escape_blue = iteration_blue[(y*config.resolutionX + x)];
				bx = x;
				by = y;
			}
		}
	}
	printf("Highest Escape E(r): %d at %d, %d\n", highest_escape_red, rx, ry);
	printf("Highest Escape E(g): %d at %d, %d\n", highest_escape_green, gx, gy);
	printf("Highest Escape E(b): %d at %d, %d\n", highest_escape_blue, bx, by);


	//std::vector<std::thread> writeT(config.threadMax);

	/*for (int sub = 0; sub < config.threadMax; ++sub)
	{
		int y_init = (config.resolutionY / config.threadMax * sub);
		int y_fin = (config.resolutionY / config.threadMax * (sub + 1));
		int x_init = 0;
		int x_fin = (config.resolutionX);

		printf("Print Thread %d initialized with parameters px:[%4d,%4d], py:[%4d,%4d]\n", sub,
		x_init, x_fin,
		y_init, y_fin
		);

		writeT[sub] = std::thread(&F_Buddhabrot::write_region, this, x_init, y_init, x_fin, y_fin);
	}

	for (int i = 0; i < config.threadMax; ++i)
	{
		writeT[i].join();
	}*/

	// each array is increased by 1 due to direct indexing, otherwise highest will index out of bounds
	uint64_t *rcount = new uint64_t[highest_escape_red + 1];
	uint64_t *gcount = new uint64_t[highest_escape_green + 1];
	uint64_t *bcount = new uint64_t[highest_escape_blue + 1];

	for (uint64_t i = 0; i < highest_escape_red + 1; i++)
	{
		rcount[i] = 0;
	}

	for (uint64_t i = 0; i < highest_escape_green + 1; i++)
	{
		gcount[i] = 0;
	}

	for (uint64_t i = 0; i < highest_escape_blue + 1; i++)
	{
		bcount[i] = 0;
	}

	// count
	for (int y = 0; y < config.resolutionY; ++y)
	{
		for (int x = 0; x < config.resolutionX; ++x)
		{
			//++rcount[calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0]];
			//++gcount[calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1]];
			//++bcount[calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2]];
			++rcount[iteration_red[y*config.resolutionX + x]];
			++gcount[iteration_green[y*config.resolutionX + x]];
			++bcount[iteration_blue[y*config.resolutionX + x]];
		}
	}

	// cumulative
	for (uint64_t i = 1; i < highest_escape_red + 1; ++i)
	{
		rcount[i] = rcount[i] + rcount[i - 1];
	}

	for (uint64_t i = 1; i < highest_escape_green + 1; ++i)
	{
		gcount[i] = gcount[i] + gcount[i - 1];
	}

	for (uint64_t i = 1; i < highest_escape_blue + 1; ++i)
	{
		bcount[i] = bcount[i] + bcount[i - 1];
	}

	double divisor = (config.resolutionX * config.resolutionY) - 1;
	for (int y = 0; y < config.resolutionY; ++y)
	{
		for (int x = 0; x < config.resolutionX; ++x)
		{
			//calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0] = (int)(((double)(rcount[calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0]] - rcount[0]) / divisor) * (255 - 2) + 1);
			//calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1] = (int)(((double)(gcount[calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1]] - gcount[0]) / divisor) * (255 - 2) + 1);
			//calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2] = (int)(((double)(bcount[calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2]] - bcount[0]) / divisor) * (255 - 2) + 1);
		
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0] = (unsigned char)(((rcount[iteration_red[(y*config.resolutionX + x)]] - rcount[0]) / divisor) * (255 - 2) + 1);
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1] = (unsigned char)(((gcount[iteration_green[(y*config.resolutionX + x)]] - gcount[0]) / divisor) * (255 - 2) + 1);
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2] = (unsigned char)(((bcount[iteration_blue[(y*config.resolutionX + x)]] - bcount[0]) / divisor) * (255 - 2) + 1);
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 3] = 255;

		}
	}

	delete[] rcount;
	delete[] gcount;
	delete[] bcount;

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("Write Buffer Time %d milliseconds\n",
		elasped
	);
}

void F_Buddhabrot::write_region(int x_init, int y_init, int x_fin, int y_fin)
{
	for (int y = y_init; y < y_fin; y++)
	{
		for (int x = x_init; x < x_fin; x++)
		{
			/*calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0] = int(255 * (sqrt(iteration_red[y*config.resolutionX + x]) / sqrt(highest_escape_red)));
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1] = int(255 * (sqrt(iteration_green[y*config.resolutionX + x]) / sqrt(highest_escape_green)));
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2] = int(255 * (sqrt(iteration_blue[y*config.resolutionX + x]) / sqrt(highest_escape_blue)));
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 3] = 255;*/
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0] = int(255 * (((double)iteration_red[y*config.resolutionX + x]) / (highest_escape_red)));
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1] = int(255 * (((double)iteration_green[y*config.resolutionX + x]) / (highest_escape_green)));
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2] = int(255 * (((double)iteration_blue[y*config.resolutionX + x]) / (highest_escape_blue)));
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 3] = 255;

		}
	}
}

void F_Buddhabrot::filter_histogram()
{

}
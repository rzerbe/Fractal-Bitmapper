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

#include "F_Mandelbrot.h"
#include "Calculation.h"
#include "Configuration.h"

// Mandelbrot

F_Mandelbrot::F_Mandelbrot(Calculation_Data &calc, Configuration &conf) 
	: calculation(calc), config(conf)
{

}

F_Mandelbrot::~F_Mandelbrot()
{
	
}

void F_Mandelbrot::fractal_render(int output)
{
	auto begin = std::chrono::high_resolution_clock::now();

	std::vector<std::thread> fractalT(config.threadMax);
	// Mode = 0 : Rectangular Subdivison
	if (config.executionMode == 0)
	{
		for (int sub = 0; sub < config.threadMax; ++sub)
		{
			int y_init = (config.resolutionY / config.threadMax * sub) * config.SSAA;
			int y_fin = (config.resolutionY / config.threadMax * (sub + 1)) * config.SSAA;
			int x_init = 0;
			int x_fin = (config.resolutionX) * config.SSAA;

			if (output == 1)
			{
				printf("Thread %d initialized with parameters px:[%4d,%4d], py:[%4d,%4d]\n", sub,
					x_init, x_fin,
					y_init, y_fin
				);
			}
			switch (config.instructionSet)
			{
			case 0:
				fractalT[sub] = std::thread(&F_Mandelbrot::avx_fractal_64, this, x_init, y_init, x_fin, y_fin, sub, config.threadMax);
				break;
			case 1:
				fractalT[sub] = std::thread(&F_Mandelbrot::avx_fractal_32, this, x_init, y_init, x_fin, y_fin, sub, config.threadMax);
				break;
			case 2:
				fractalT[sub] = std::thread(&F_Mandelbrot::c_fractal_64, this, x_init, y_init, x_fin, y_fin, sub, config.threadMax);
				break;
			}
		}
	}
	// Mode = 1 : Square Subdivison
	else if (config.executionMode == 1)
	{
		int thread = 0;
		for (int y = 0; y < config.threadMax / 2; ++y) // Starting y subdivison
		{
			for (int x = 0; x < (config.threadMax / 2); ++x) // Starting x subdivison
			{
				if (x == config.threadMax / 2)
				{
					switch (config.instructionSet)
					{
					case 0:
						fractalT[thread] = std::thread(&F_Mandelbrot::avx_fractal_64, this, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 1:
						fractalT[thread] = std::thread(&F_Mandelbrot::avx_fractal_32, this, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 2:
						fractalT[thread] = std::thread(&F_Mandelbrot::c_fractal_64, this, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					}
					if (output == 1)
					{
						printf("Thread %d initialized with parameters px:[%4d,%4d], py:[%4d,%4d]\n", thread,
							(config.resolutionX*x) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))),
							(config.resolutionY*y) / (config.threadMax / 2), (config.resolutionY*(y + 1)) / (config.resolutionY*(y + 1)) / (config.threadMax / 2)
						);
					}
					thread++;
				}
				else if (y == config.threadMax / 2)
				{
					switch (config.instructionSet)
					{
					case 0:
						fractalT[thread] = std::thread(&F_Mandelbrot::avx_fractal_64, this, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 1:
						fractalT[thread] = std::thread(&F_Mandelbrot::avx_fractal_32, this, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 2:
						fractalT[thread] = std::thread(&F_Mandelbrot::c_fractal_64, this, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					}
					if (output == 1)
					{
						printf("Thread %d initialized with parameters px:[%4d,%4d], py:[%4d,%4d]\n", thread,
							(config.resolutionX*x) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))),
							(config.resolutionY*y) / (config.threadMax / 2), (config.resolutionY*(y + 1)) / (config.threadMax / 2)
						);

					}
					thread++;
				}
				else
				{
					switch (config.instructionSet)
					{
					case 0:
						fractalT[thread] = std::thread(&F_Mandelbrot::avx_fractal_64, this, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 1:
						fractalT[thread] = std::thread(&F_Mandelbrot::avx_fractal_32, this, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 2:
						fractalT[thread] = std::thread(&F_Mandelbrot::c_fractal_64, this, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					}
					if (output == 1)
					{
						printf("Thread %d initialized with parameters px:[%4d,%4d], py:[%4d,%4d]\n", thread,
							(config.resolutionX*x) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))),
							(config.resolutionY*y) / (config.threadMax / 2), (config.resolutionY*(y + 1)) / (config.threadMax / 2)
						);
					}
					thread++;
				}
			}
		}
	}
	// Mode = 2 : Work Queue
	else if (config.executionMode == 2)
	{

	}
	for (int i = 0; i < config.threadMax; ++i)
	{
		fractalT[i].join();
	}

	// Post Processing
	downsampler();
	write_buffer();


	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("Fractal Generate Time %d\n",
		elasped
	);
}

void F_Mandelbrot::avx_fractal_32(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads)
{
	auto begin = std::chrono::high_resolution_clock::now();

	__m256 ymm_dx = _mm256_set1_ps((float)((config.xmax - config.xmin) / config.bufferX));
	__m256 ymm_dy = _mm256_set1_ps((float)((config.ymax - config.ymin) / config.bufferY));
	__m256 ymm_wx = _mm256_set1_ps((float)config.xmin);
	__m256 ymm_wy = _mm256_set1_ps((float)config.ymin);
	__m256 ymm_inc = _mm256_set1_ps(1.0f);
	__m256 ymm_rad = _mm256_set1_ps(4.0f);
	__m256 ymm_j = _mm256_set1_ps((float)j_init);

	for (int j = j_init; j < j_fin; ++j)
	{
		__m256 ymm_i = _mm256_set_ps(0.0f + i_init, 1.0f + i_init, 2.0f + i_init, 3.0f + i_init, 4.0f + i_init, 5.0f + i_init, 6.0f + i_init, 7.0f + i_init);
		for (int i = i_init; i < i_fin; i += 8)
		{
			// Rendering shortcut
			if (calculation.screenOptimization == true)
			{
				// Check : No update
				if (calculation.updatePixel[j*config.resolutionX + i + 0] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 1] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 2] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 3] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 4] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 5] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 6] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 7] == false)
				{
					ymm_i = _mm256_add_ps(ymm_i, ymm_rad);	// i = i + 4
					ymm_i = _mm256_add_ps(ymm_i, ymm_rad);  // i = i + 4
					continue;
				}
			}
			unsigned int iter = 0;	// Counter for highest iteration
			int test = 0;	// bits [7:0] used in comparison

			__m256 ymm_cr = _mm256_mul_ps(ymm_i, ymm_dx);					// cr[] = (i) * (dx)
			ymm_cr = _mm256_add_ps(ymm_cr, ymm_wx);							// cr[] = (i*dx) + (x)
			__m256 ymm_ci = _mm256_mul_ps(ymm_j, ymm_dy);					// ci[] = (j) * (dy)
			ymm_ci = _mm256_add_ps(ymm_ci, ymm_wy);							// ci[] = (j*dy) + (y)
			__m256 ymm_iter = _mm256_xor_ps(ymm_dx, ymm_dx);				// iter[] = 0
			__m256 ymm_zr = ymm_iter;										// xi[] = 0
			__m256 ymm_zi = ymm_iter;										// yi[] = 0

			do
			{
				__m256 ymm_zrzr = _mm256_mul_ps(ymm_zr, ymm_zr);			// zrzr[] = (zr[]) * (zr[])
				__m256 ymm_zizi = _mm256_mul_ps(ymm_zi, ymm_zi);			// zizi[] = (zi[]) * (zi[])
				__m256 ymm_temp = _mm256_add_ps(ymm_zrzr, ymm_zizi);		// temp[] = (zr[]*zr[]) + (zi[]*zi[])

																			// zr*zr+zi*zi < 4
				ymm_temp = _mm256_cmp_ps(ymm_temp, ymm_rad, _CMP_LT_OQ);	// temp[] = {0^32 | 1^32}
				test = _mm256_movemask_ps(ymm_temp) & 255;					// test =	(0^8  | 1^8) AND 11111111
				ymm_temp = _mm256_and_ps(ymm_temp, ymm_inc);				// temp[] = {0.0f | 1.0f}
				ymm_iter = _mm256_add_ps(ymm_iter, ymm_temp);				// iter[] = (iter[]) + (temp[])

				ymm_temp = _mm256_mul_ps(ymm_zr, ymm_zi);					// temp[] = (zr[]) * (zi[])
				ymm_zr = _mm256_sub_ps(ymm_zrzr, ymm_zizi);					// zr[] = (zr[]*zr[]) - (zi[]*zi[])
				ymm_zr = _mm256_add_ps(ymm_zr, ymm_cr);						// zr[] = (zr[]*zr[]-zi[]*zi[]) + (cr)
				ymm_zi = _mm256_add_ps(ymm_temp, ymm_temp);					// zi[] = (xi*yi) + (xi*yi)
				ymm_zi = _mm256_add_ps(ymm_zi, ymm_ci);						// zi[] = (2*xi*yi) + (ci)	

				++iter;
			} while ((test != 0) && (iter < config.maxIter));

			// Discard excess render past border
			int top = (i + 7) < config.bufferX ? 8 : config.bufferX & 7;
			for (int k = 0; k < top; ++k)
			{
				calculation.escapeBufferSuperSampling[i + k + j*config.bufferX] = ymm_iter.m256_f32[top - k - 1];
			}

			ymm_i = _mm256_add_ps(ymm_i, ymm_rad);	// i = i + 4
			ymm_i = _mm256_add_ps(ymm_i, ymm_rad);  // i = i + 4
		}
		ymm_j = _mm256_add_ps(ymm_j, ymm_inc);		// j++
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("AVX_32 Thread #%d finished in %d milliseconds \n",
		threadID, elasped
	);
}

void F_Mandelbrot::avx_fractal_64(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads)
{
	auto begin = std::chrono::high_resolution_clock::now();

	__m256d ymm_dx = _mm256_set1_pd(((config.xmax - config.xmin) / config.bufferX));
	__m256d ymm_dy = _mm256_set1_pd(((config.ymax - config.ymin) / config.bufferY));
	__m256d ymm_wx = _mm256_set1_pd(config.xmin);
	__m256d ymm_wy = _mm256_set1_pd(config.ymin);
	__m256d ymm_inc = _mm256_set1_pd(1.0);
	__m256d ymm_rad = _mm256_set1_pd(4.0);
	__m256d ymm_j = _mm256_set1_pd(j_init);

	for (int j = j_init; j < j_fin; j += 1)
	{
		__m256d ymm_i = _mm256_set_pd(0.0f + i_init, 1.0f + i_init, 2.0f + i_init, 3.0f + i_init);
		for (int i = i_init; i < i_fin; i += 4)
		{
			// Rendering shortcut
			if (calculation.screenOptimization == true)
			{
				if (calculation.updatePixel[j*config.resolutionX + i + 0] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 1] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 2] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 3] == false)
				{
					ymm_i = _mm256_add_pd(ymm_i, ymm_rad);	// i = i + 4
					continue;
				}
			}

			unsigned int iter = 0;	// Counter for highest iteration
			int test = 0;	// bits [3:0] used in comparison

			__m256d ymm_cr = _mm256_mul_pd(ymm_i, ymm_dx);					// cr[] = (i) * (dx)
			ymm_cr = _mm256_add_pd(ymm_cr, ymm_wx);							// cr[] = (i*dx) + (x)
			__m256d ymm_ci = _mm256_mul_pd(ymm_j, ymm_dy);					// ci[] = (j) * (dy)
			ymm_ci = _mm256_add_pd(ymm_ci, ymm_wy);							// ci[] = (j*dy) + (y)
			__m256d ymm_iter = _mm256_xor_pd(ymm_dx, ymm_dx);				// iter[] = 0
			__m256d ymm_zr = ymm_iter;										// xi[] = 0
			__m256d ymm_zi = ymm_iter;										// yi[] = 0

			do
			{
				__m256d ymm_zrzr = _mm256_mul_pd(ymm_zr, ymm_zr);			// zrzr[] = (zr[]) * (zr[])
				__m256d ymm_zizi = _mm256_mul_pd(ymm_zi, ymm_zi);			// zizi[] = (zi[]) * (zi[])
				__m256d ymm_temp = _mm256_add_pd(ymm_zrzr, ymm_zizi);		// temp[] = (zr[]*zr[]) + (zi[]*zi[])

																			// zr*zr+zi*zi < 4
				ymm_temp = _mm256_cmp_pd(ymm_temp, ymm_rad, _CMP_LT_OQ);	// temp[] = {0^64 | 1^64}
				test = _mm256_movemask_pd(ymm_temp) & 15;					// test =	(0^4  | 1^4) AND 1111
				ymm_temp = _mm256_and_pd(ymm_temp, ymm_inc);				// temp[] = {0.0f | 1.0f}
				ymm_iter = _mm256_add_pd(ymm_iter, ymm_temp);				// iter[] = (iter[]) + (temp[])

				ymm_temp = _mm256_mul_pd(ymm_zr, ymm_zi);					// temp[] = (zr[]) * (zi[])
				ymm_zr = _mm256_sub_pd(ymm_zrzr, ymm_zizi);					// zr[] = (zr[]*zr[]) - (zi[]*zi[])
				ymm_zr = _mm256_add_pd(ymm_zr, ymm_cr);						// zr[] = (zr[]*zr[]-zi[]*zi[]) + (cr)
				ymm_zi = _mm256_add_pd(ymm_temp, ymm_temp);					// zi[] = (xi*yi) + (xi*yi)
				ymm_zi = _mm256_add_pd(ymm_zi, ymm_ci);						// zi[] = (2*xi*yi) + (ci)	

				++iter;
			} while ((test != 0) && (iter < config.maxIter));


			// Discard excess render past border
			int top = (i + 3) < config.bufferX ? 4 : config.bufferX & 3;
			for (int k = 0; k < top; ++k)
			{
				calculation.escapeBufferSuperSampling[i + k + j*config.bufferX] = ymm_iter.m256d_f64[top - k - 1];
			}

			ymm_i = _mm256_add_pd(ymm_i, ymm_rad);	// i = i + 4
		}
		ymm_j = _mm256_add_pd(ymm_j, ymm_inc);		// j++
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("AVX_64 Thread #%d finished in %d milliseconds \n",
		threadID, elasped
	);
}

void F_Mandelbrot::c_fractal_64(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads)
{
	auto begin = std::chrono::high_resolution_clock::now();
	double zr;
	double zi;
	double zr_prev;

	double xWindow = config.xmax - config.xmin;
	double yWindow = config.ymax - config.ymin;

	int m;
	double dub;
	int k;
	double ci, cr, q, logzMag, zMagSqr, potential;
	bool shortcut;

	for (int j = j_init; j < j_fin; ++j) // y axis
	{
		/*//give percentage progress
		m = j % (h / 100);
		if (m == 0)
		{
		std::cout << "Thread: " << threadID << " Computation " << (int)(((double)(j - j_init) / (j_fin - j_init) * 100)) << "% complete.\n";
		}*/
		ci = config.ymin + ((double)j) * (yWindow) / config.bufferY;

		for (int i = i_init; i < i_fin; ++i) // x axis
		{
			shortcut = false;
			cr = config.xmin + ((double)i) * (xWindow) / config.bufferX;

			//check whether point lies in main cardioid

			if (calculation.screenOptimization == 1)
			{
				q = (cr - 0.25)*(cr - 0.25) + ci*ci;
				if (q*(q + cr - 0.25) - 0.25*ci*ci < 0)
				{
					calculation.escapeBufferSuperSampling[j*config.bufferX + i] = config.maxIter;
					calculation.magnitude[j*config.bufferX + i] = INFINITY;
					shortcut = true;
				}
				//check whether point lies in period-2 bulb
				else if ((cr + 1)*(cr + 1) + ci*ci - 0.0625 < 0)
				{
					calculation.escapeBufferSuperSampling[j*config.bufferX + i] = config.maxIter;
					calculation.magnitude[j*config.bufferX + i] = INFINITY;
					shortcut = true;
				}
			}

			//if point lies within main cardioid or bulb, skip computation

			if (!shortcut)
			{
				zr = 0;
				zi = 0;

				if (calculation.screenOptimization == true)
				{
					if (calculation.updatePixel[j*config.bufferX + i] == false)
					{
						shortcut = true;
					}
				}

				if (shortcut == false)
				{
					for (k = 1; k < config.maxIter; ++k)
					{
						zr_prev = zr;
						zr = zr * zr - zi * zi + cr;
						zi = 2 * zr_prev * zi + ci;

						zMagSqr = zr * zr + zi * zi;
						if (zMagSqr > 4.0)
						{
							break;
						}
					}
					calculation.escapeBufferSuperSampling[j*config.bufferX + i] = k;
					calculation.magnitude[j*config.bufferX + i] = zMagSqr;
				}
			}
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("Standard Thread #%d finished in %d milliseconds \n",
		threadID, elasped
	);
}

void F_Mandelbrot::write_pixel(int x, int y)
{
	//colorDraw is the color we will draw with
	//color1 and color2 are the colors that need to be linearly interpolated to get colorDraw
	//palette stores the rgb of all of our hardcoded colors

	if (calculation.escapeBufferCPU[y*config.resolutionX + x] == config.maxIter)
	{
		calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0] = 0;
		calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1] = 0;
		calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2] = 0;
		calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 3] = 255;
	}
	else
	{
		if (config.shadingMode == 0)
		{
			unsigned char color1[3];
			unsigned char color2[3];

			double k = calculation.escapeBufferCPU[y*config.resolutionX + x];

			/*
			This code tries to generate a color based on the escapeMatrix value.  This matrix contains
			continuous values from 0 to maxIter.  Imagine the escapeMatrix values (0 to maxIter) being
			divided into intervals of size 'colorDensity'.  For instance, the first interval would then
			be 0 to 'colorDensity', second interval would be 'colorDensity' to 2*'colorDensity'.
			For each of these intervals, the color to be drawn with is determined by linearly interpolating
			the palette (using indices only up to 'colorsUsed'-1) based on the escapeMatrix value.
			Overall, this creates a cyclical coloring scheme.  The colors will repeat the further we zoom
			into the image.
			*/

			//colorDraw is the color we will draw with
			//color1 and color2 are the colors that need to be linearly interpolated to get colorDraw

			/*logzMag = abs(log(zMagSqr) / 2);
			//potential = log(logzMag / log(2)) / log(2);
			potential = log(logzMag) / log(2);
			dub = k + 1 - potential;			//this result is a double
			if (dub > maxIter || dub < -1000)
			dub = maxIter;
			escapeMatrix[j*w + i] = dub;
			write_pixel(i, j);*/

			//figure out which two colors our point should be interpolated between, based on k
			double kscaled = fmod(k, config.colorDensity) / config.colorDensity * (config.paletteNumber);
			color1[0] = config.palette[4 * (int)floor(kscaled) + 0];
			color1[1] = config.palette[4 * (int)floor(kscaled) + 1];
			color1[2] = config.palette[4 * (int)floor(kscaled) + 2];

			color2[0] = config.palette[4 * (int)floor(kscaled + 1) + 0];
			color2[1] = config.palette[4 * (int)floor(kscaled + 1) + 1];
			color2[2] = config.palette[4 * (int)floor(kscaled + 1) + 2];

			//linearly interpolate between color1 and color2, based on fractional part of k
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0] = (color2[0] - color1[0])*(kscaled - floor(kscaled)) + color1[0];
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1] = (color2[1] - color1[1])*(kscaled - floor(kscaled)) + color1[1];
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2] = (color2[2] - color1[2])*(kscaled - floor(kscaled)) + color1[2];
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 3] = 255;
		}
		else if (config.shadingMode == 1)
		{
			//Monochromatic
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0] = config.maxIter - calculation.escapeBufferCPU[y*config.resolutionX + x];
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1] = config.maxIter - calculation.escapeBufferCPU[y*config.resolutionX + x];
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2] = config.maxIter - calculation.escapeBufferCPU[y*config.resolutionX + x];
			calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 3] = 255;
		}
	}
}

void F_Mandelbrot::write_buffer()
{
	auto begin = std::chrono::high_resolution_clock::now();

	std::vector<std::thread> writeT(config.threadMax);

	for (int sub = 0; sub < config.threadMax; ++sub)
	{
		int y_init = (config.resolutionY / config.threadMax * sub);
		int y_fin = (config.resolutionY / config.threadMax * (sub + 1));
		int x_init = 0;
		int x_fin = (config.resolutionX);

		/*printf("Print Thread %d initialized with parameters px:[%4d,%4d], py:[%4d,%4d]\n", sub,
		x_init, x_fin,
		y_init, y_fin
		);*/

		writeT[sub] = std::thread(&F_Mandelbrot::write_region, this, x_init, y_init, x_fin, y_fin);
	}

	for (int i = 0; i < config.threadMax; ++i)
	{
		writeT[i].join();
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	printf("Write Buffer Time %d milliseconds\n",
		elasped
	);
}

void F_Mandelbrot::write_region(int x_init, int y_init, int x_fin, int y_fin)
{
	int colord = config.colorDensity;
	int colorn = config.paletteNumber;

	for (int y = y_init; y < y_fin; y++)
	{
		for (int x = x_init; x < x_fin; x++)
		{
			if (calculation.escapeBufferCPU[y*config.resolutionX + x] == config.maxIter)
			{
				calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0] = 0;
				calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1] = 0;
				calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2] = 0;
				calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 3] = 255;
			}
			else
				if (config.shadingMode == 0)
				{
					unsigned char color1[3];
					unsigned char color2[3];

					double k = calculation.escapeBufferCPU[y*config.resolutionX + x];

					/*
					This code tries to generate a color based on the escapeMatrix value.  This matrix contains
					continuous values from 0 to maxIter.  Imagine the escapeMatrix values (0 to maxIter) being
					divided into intervals of size 'colorDensity'.  For instance, the first interval would then
					be 0 to 'colorDensity', second interval would be 'colorDensity' to 2*'colorDensity'.
					For each of these intervals, the color to be drawn with is determined by linearly interpolating
					the palette (using indices only up to 'colorsUsed'-1) based on the escapeMatrix value.
					Overall, this creates a cyclical coloring scheme.  The colors will repeat the further we zoom
					into the image.
					*/

					//colorDraw is the color we will draw with
					//color1 and color2 are the colors that need to be linearly interpolated to get colorDraw

					/*logzMag = abs(log(zMagSqr) / 2);
					//potential = log(logzMag / log(2)) / log(2);
					potential = log(logzMag) / log(2);
					dub = k + 1 - potential;			//this result is a double
					if (dub > maxIter || dub < -1000)
					dub = maxIter;
					escapeMatrix[j*w + i] = dub;
					write_pixel(i, j);*/

					//figure out which two colors our point should be interpolated between, based on k

					//double kscaled = fmod(k, config.colorDensity) / config.colorDensity * (config.paletteNumber);
					double kscaled = (k - colord * (int)(k / colord)) / colord * colorn;

					color1[0] = config.palette[4 * (int)(kscaled)+0];
					color1[1] = config.palette[4 * (int)(kscaled)+1];
					color1[2] = config.palette[4 * (int)(kscaled)+2];

					color2[0] = config.palette[4 * (int)(kscaled + 1) + 0];
					color2[1] = config.palette[4 * (int)(kscaled + 1) + 1];
					color2[2] = config.palette[4 * (int)(kscaled + 1) + 2];


					//linearly interpolate between color1 and color2, based on fractional part of k
					calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0] = (color2[0] - color1[0])*(kscaled - (int)(kscaled)) + color1[0];
					calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1] = (color2[1] - color1[1])*(kscaled - (int)(kscaled)) + color1[1];
					calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2] = (color2[2] - color1[2])*(kscaled - (int)(kscaled)) + color1[2];
					calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 3] = 255;

				}
				else if (config.shadingMode == 1)
				{
					//Monochromatic
					calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 0] = config.maxIter - calculation.escapeBufferCPU[y*config.resolutionX + x];
					calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 1] = config.maxIter - calculation.escapeBufferCPU[y*config.resolutionX + x];
					calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 2] = config.maxIter - calculation.escapeBufferCPU[y*config.resolutionX + x];
					calculation.screenBufferCPU[4 * (y*config.resolutionX + x) + 3] = 255;
				}
		}
	}
}

void F_Mandelbrot::normalize()
{
	/*logzMag = abs(log(zMagSqr) / 2);
	//potential = log(logzMag / log(2)) / log(2);
	potential = log(logzMag) / log(2);
	dub = k + 1 - potential;			//this result is a double
	if (dub > maxIter || dub < -1000)
	dub = maxIter;
	escapeMatrix[j*config.bufferX + i] = dub;
	*/

	auto begin = std::chrono::high_resolution_clock::now();
	// Sub-Pixel
	for (int y = 0; y < config.bufferY; ++y)
	{
		for (int x = 0; x < config.bufferX; ++x)
		{

		}
	}
}

void F_Mandelbrot::downsampler()
{
	/*logzMag = abs(log(zMagSqr) / 2);
	//potential = log(logzMag / log(2)) / log(2);
	potential = log(logzMag) / log(2);
	dub = k + 1 - potential;			//this result is a double
	if (dub > maxIter || dub < -1000)
	dub = maxIter;
	escapeMatrix[j*config.bufferX + i] = dub;
	*/

	auto begin = std::chrono::high_resolution_clock::now();

	std::vector<std::thread> writeT(config.threadMax);

	for (int sub = 0; sub < config.threadMax; ++sub)
	{
		int j_init = (config.resolutionY / config.threadMax * sub);
		int j_fin = (config.resolutionY / config.threadMax * (sub + 1));
		int i_init = 0;
		int i_fin = (config.resolutionX);

		/*printf("Downsample Thread %d initialized with parameters px:[%4d,%4d], py:[%4d,%4d]\n", sub,
			i_init, i_fin,
			j_init, j_fin
		);*/

		writeT[sub] = std::thread(&F_Mandelbrot::downsample_region, this, i_init, j_init, i_fin, j_fin);
	}

	for (int i = 0; i < config.threadMax; ++i)
	{
		writeT[i].join();
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	printf("Downsampling Time %d milliseconds\n",
		elasped
	);
}

void F_Mandelbrot::downsample_region(int i_init, int j_init, int i_fin, int j_fin)
{
	// Pixel
	int index;
	int smallestIndex;
	double min;
	double norm;
	for (int j = j_init; j < j_fin; ++j)
	{
		for (int i = i_init; i < i_fin; ++i)
		{

			min = config.maxIter;
			smallestIndex = ((j * config.SSAA + 0)*config.resolutionX * config.SSAA) + (i * config.SSAA + 0);
			// Sub-Pixel
			for (int y = 0; y < config.SSAA; ++y)
			{
				for (int x = 0; x < config.SSAA; ++x)
				{

					index = ((j * config.SSAA + y)*config.resolutionX * config.SSAA) + (i * config.SSAA + x);
					if (calculation.escapeBufferSuperSampling[index] < min)
					{
						smallestIndex = index;
						min = calculation.escapeBufferSuperSampling[index];
					}
				}
			}

			if (config.smoothShading == true)
			{
				double logzmag = abs(log(calculation.magnitude[smallestIndex])) / 2;
				double potential = log(logzmag / log(2)) / log(2);
				norm = min + 1 - potential;
				if (norm > config.maxIter || norm < -1000 || min == config.maxIter - 1)
				{
					norm = config.maxIter;
				}

				calculation.escapeBufferCPU[j*config.resolutionX + i] = norm;
			}
			else
			{
				calculation.escapeBufferCPU[j*config.resolutionX + i] = min;
			}
		}
	}
}
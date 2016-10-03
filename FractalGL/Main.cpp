// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <ostream>
#include <iomanip>  
#include <sstream>
#include <queue>

// CPU
#include <chrono>
#include <thread>
#include <immintrin.h>
#include <intrin.h>

// Renderer
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>

// Program Structure
#include "Configuration.h"
#include "Calculation.h"

////////////////////////////////////////
// Program Data
Configuration config;
Calcuation_Data calculation;
__declspec(align(64)) Calculation_Thread calculationThread[32];

// Graphics
GLuint gl_PBO, gl_Tex, gl_Shader;
GLFWwindow* window;
////////////////////////////////////////

void displayFunc();
void pollCommandPrompt();

// Mandelbrot
void renderOptimizer(int iterations, int delta)
{
	// Increasing number of iterations
	if (delta > 0)
	{
		for (int j = 0; j < config.resolutionY; ++j)
		{
			for (int i = 0; i < config.resolutionX; ++i)
			{
				if (calculation.escapeBufferCPU[(j*config.resolutionX + i)] == calculation.prevMaxIter)
				{
					calculation.updatePixel[(j*config.resolutionX + i)] = true;
				}
				else
				{
					calculation.updatePixel[(j*config.resolutionX + i)] = false;
				}
			}
		}
	}
	// Decreasing number of iterations
	if (delta < 0)
	{
		for (int j = 0; j < config.resolutionY; j++)
		{
			for (int i = 0; i < config.resolutionX; i++)
			{
				if (calculation.escapeBufferCPU[(j*config.resolutionX + i)] > config.maxIter)
				{
					calculation.updatePixel[(j*config.resolutionX + i)] = true;
				}
				else
				{
					calculation.updatePixel[(j*config.resolutionX + i)] = false;
				}
			}
		}
	}
}

void writePixel(int x, int y)
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
			writePixel(i, j);*/

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

void avx_fractal(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads)
{
	auto begin = std::chrono::high_resolution_clock::now();
	bool shortcut = false;
	float constants[] = { (config.xmax - config.xmin) / config.resolutionX, (config.ymax - config.ymin) / config.resolutionY, config.xmin, config.ymin, 1.0f, 4.0f, j_init };
	__m256 ymm0 = _mm256_broadcast_ss(constants);   // all dx
	__m256 ymm1 = _mm256_broadcast_ss(constants + 1); // all dy
	__m256 ymm2 = _mm256_broadcast_ss(constants + 2); // all x1
	__m256 ymm3 = _mm256_broadcast_ss(constants + 3); // all y1
	__m256 ymm4 = _mm256_broadcast_ss(constants + 4); // all 1's (iter increments)
	__m256 ymm5 = _mm256_broadcast_ss(constants + 5); // all 4's (comparisons)
	__m256 ymm6 = _mm256_broadcast_ss(constants + 6); // set to starting y position

	float incr[8] = { 0.0f + i_init,1.0f + i_init,2.0f + i_init,3.0f + i_init,4.0f + i_init,5.0f + i_init,6.0f + i_init, 7.0f + i_init }; // used to reset the i position when j increases

	for (int j = j_init; j < j_fin; ++j)
	{
		__m256 ymm7 = _mm256_set_ps(incr[0], incr[1], incr[2], incr[3], incr[4], incr[5], incr[6], incr[7]);  // i counter set to 0,1,2,..,7
		for (int i = i_init; i < i_fin; i += 8)
		{
			// Rendering shortcut
			if (calculation.screenOptimization == true)
			{
				if (calculation.updatePixel[j*config.resolutionX + i + 0] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 1] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 2] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 3] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 4] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 5] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 6] == false &&
					calculation.updatePixel[j*config.resolutionX + i + 7] == false)
				{
					shortcut = true;
				}
			}

			if (shortcut == false)
			{
				int test = 0;
				unsigned int iter = 0;

				__m256 ymm8 = _mm256_mul_ps(ymm7, ymm0);  // x0 = (i+k)*dx 
				ymm8 = _mm256_add_ps(ymm8, ymm2);         // x0 = x1+(i+k)*dx
				__m256 ymm9 = _mm256_mul_ps(ymm6, ymm1);  // y0 = j*dy
				ymm9 = _mm256_add_ps(ymm9, ymm3);         // y0 = y1+j*dy
				__m256 ymm10 = _mm256_xor_ps(ymm0, ymm0);  // zero out iteration counter
				__m256 ymm11 = ymm10, ymm12 = ymm10;        // set initial xi=0, yi=0

				do
				{
					__m256 ymm13 = _mm256_mul_ps(ymm11, ymm11); // xi*xi
					__m256 ymm14 = _mm256_mul_ps(ymm12, ymm12); // yi*yi
					__m256 ymm15 = _mm256_add_ps(ymm13, ymm14); // xi*xi+yi*yi

																// xi*xi+yi*yi < 4 in each slot
					ymm15 = _mm256_cmp_ps(ymm15, ymm5, _CMP_LT_OQ);
					// now ymm15 has all 1s in the non overflowed locations
					test = _mm256_movemask_ps(ymm15) & 255;      // lower 8 bits are comparisons
					ymm15 = _mm256_and_ps(ymm15, ymm4);
					// get 1.0f or 0.0f in each field as counters
					// counters for each pixel iteration
					ymm10 = _mm256_add_ps(ymm10, ymm15);

					ymm15 = _mm256_mul_ps(ymm11, ymm12);        // xi*yi 
					ymm11 = _mm256_sub_ps(ymm13, ymm14);        // xi*xi-yi*yi
					ymm11 = _mm256_add_ps(ymm11, ymm8);         // xi <- xi*xi-yi*yi+x0 done!
					ymm12 = _mm256_add_ps(ymm15, ymm15);        // 2*xi*yi
					ymm12 = _mm256_add_ps(ymm12, ymm9);         // yi <- 2*xi*yi+y0	

					++iter;
				} while ((test != 0) && (iter < config.maxIter));

				// write only where needed
				int top = (i + 7) < config.resolutionX ? 8 : config.resolutionX & 7;
				for (int k = 0; k < top; ++k)
				{
					if (ymm10.m256_f32[top - k - 1] == (float)config.maxIter)
					{
						calculation.updatePixel[j*config.resolutionX + i + k] = true;
						calculation.escapeBufferCPU[j*config.resolutionX + i + k] = (double)config.maxIter;
						writePixel(i + k, j);
					}
					else
					{
						calculation.escapeBufferCPU[i + k + j*config.resolutionX] = ymm10.m256_f32[top - k - 1];
						writePixel(i + k, j);
					}
				}
			}
			// next i position - increment each slot by 8
			ymm7 = _mm256_add_ps(ymm7, ymm5);
			ymm7 = _mm256_add_ps(ymm7, ymm5);
			shortcut = false;
		}
		ymm6 = _mm256_add_ps(ymm6, ymm4); // increment j counter
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("AVX_32 Thread #%d finished in %d milliseconds \n",
		threadID, elasped
		);
}

void avx_fractal_64(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads)
{
	auto begin = std::chrono::high_resolution_clock::now();
	bool shortcut = false;

	double constants[] = { (config.xmax - config.xmin) / config.resolutionX, (config.ymax - config.ymin) / config.resolutionY, config.xmin, config.ymin, 1.0, 4.0, j_init };
	__m256d ymm0 = _mm256_broadcast_sd(constants);   // all dx
	__m256d ymm1 = _mm256_broadcast_sd(constants + 1); // all dy
	__m256d ymm2 = _mm256_broadcast_sd(constants + 2); // all x1
	__m256d ymm3 = _mm256_broadcast_sd(constants + 3); // all y1
	__m256d ymm4 = _mm256_broadcast_sd(constants + 4); // all 1's (iter increments)
	__m256d ymm5 = _mm256_broadcast_sd(constants + 5); // all 4's (comparisons)
	__m256d ymm6 = _mm256_broadcast_sd(constants + 6); // set to starting y position

	double incr[8] = { 0.0 + i_init,1.0 + i_init,2.0 + i_init,3.0 + i_init }; // used to reset the i position when j increases

	for (int j = j_init; j < j_fin; j += 1)
	{
		__m256d ymm7 = _mm256_set_pd(incr[0], incr[1], incr[2], incr[3]);  // i counter set to 0,1,2,3
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
					shortcut = true;
				}
			}

			if (shortcut == false)
			{
				__m256d ymm8 = _mm256_mul_pd(ymm7, ymm0);  // x0 = (i+k)*dx 
				ymm8 = _mm256_add_pd(ymm8, ymm2);         // x0 = x1+(i+k)*dx
				__m256d ymm9 = _mm256_mul_pd(ymm6, ymm1);  // y0 = j*dy
				ymm9 = _mm256_add_pd(ymm9, ymm3);         // y0 = y1+j*dy
				__m256d ymm10 = _mm256_xor_pd(ymm0, ymm0);  // zero out iteration counter
				__m256d ymm11 = ymm10, ymm12 = ymm10;        // set initial xi=0, yi=0

				int test = 0;
				unsigned int iter = 0;
				do
				{
					__m256d ymm13 = _mm256_mul_pd(ymm11, ymm11); // xi*xi
					__m256d ymm14 = _mm256_mul_pd(ymm12, ymm12); // yi*yi
					__m256d ymm15 = _mm256_add_pd(ymm13, ymm14); // xi*xi+yi*yi

																 // xi*xi+yi*yi < 4 in each slot
					ymm15 = _mm256_cmp_pd(ymm15, ymm5, _CMP_LT_OQ);
					// now ymm15 has all 1s in the non overflowed locations
					test = _mm256_movemask_pd(ymm15) & 255;      // lower 8 bits are comparisons
					ymm15 = _mm256_and_pd(ymm15, ymm4);
					// get 1.0f or 0.0f in each field as counters
					// counters for each pixel iteration
					ymm10 = _mm256_add_pd(ymm10, ymm15);

					ymm15 = _mm256_mul_pd(ymm11, ymm12);        // xi*yi 
					ymm11 = _mm256_sub_pd(ymm13, ymm14);        // xi*xi-yi*yi
					ymm11 = _mm256_add_pd(ymm11, ymm8);         // xi <- xi*xi-yi*yi+x0 done!
					ymm12 = _mm256_add_pd(ymm15, ymm15);        // 2*xi*yi
					ymm12 = _mm256_add_pd(ymm12, ymm9);         // yi <- 2*xi*yi+y0	

					++iter;
				} while ((test != 0) && (iter < config.maxIter));


				// write only where needed
				int top = (i + 3) < config.resolutionX ? 4 : config.resolutionX & 3;
				for (int k = 0; k < top; ++k)
				{
					calculation.escapeBufferCPU[i + k + j*config.resolutionX] = ymm10.m256d_f64[top - k - 1];
					writePixel(i + k, j);
				}
			}

			// next i position - increment each slot by 4
			ymm7 = _mm256_add_pd(ymm7, ymm5);
			shortcut = false;
		}
		ymm6 = _mm256_add_pd(ymm6, ymm4); // increment j counter
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("AVX_64 Thread #%d finished in %d milliseconds \n",
		threadID, elasped
		);
}

void fractal(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads)
{
	auto begin = std::chrono::high_resolution_clock::now();
	double zrCurrent;
	double ziCurrent;
	double zrPrevious;
	double ziPrevious;

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
		ci = config.ymin + ((double)j) * (yWindow) / config.resolutionY;

		for (int i = i_init; i < i_fin; ++i) // x axis
		{
			shortcut = false;
			cr = config.xmin + ((double)i) * (xWindow) / config.resolutionX;

			//check whether point lies in main cardioid
			/*q = (cr - 0.25)*(cr - 0.25) + ci*ci;
			if (q*(q + cr - 0.25) - 0.25*ci*ci < 0)
			{
			escapeMatrix[j*config.resolutionX + i] = maxIter;
			writePixel(i, j);
			shortcut = true;
			}
			//check whether point lies in period-2 bulb
			else if ((cr + 1)*(cr + 1) + ci*ci - 0.0625 < 0)
			{
			escapeMatrix[j*config.resolutionX + i] = maxIter;
			writePixel(i, j);
			shortcut = true;
			}*/

			//if point lies within main cardioid or bulb, skip computation

			if (!shortcut)
			{
				zrPrevious = 0;
				ziPrevious = 0;

				if (calculation.screenOptimization == true)
				{
					if (calculation.updatePixel[j*config.resolutionX + i] == false)
					{
						shortcut = true;
					}
				}

				if (shortcut == false)
				{
					for (k = 1; k < config.maxIter; ++k)
					{
						//Optimized
						zrCurrent = zrPrevious * zrPrevious - ziPrevious * ziPrevious + cr;
						ziCurrent = 2 * zrPrevious * ziPrevious + ci;
						zMagSqr = zrCurrent * zrCurrent + ziCurrent * ziCurrent;

						zrPrevious = zrCurrent;
						ziPrevious = ziCurrent;

						if (zMagSqr > 4)		//did some optimizations in this area
						{
							break;
						}
					}
					//Non-convergence (k was stored as -2^31 - 1 if not handled)
					if (k == config.maxIter)
					{
						calculation.updatePixel[j*config.resolutionX + i] = true;
						calculation.escapeBufferCPU[j*config.resolutionX + i] = config.maxIter;
						writePixel(i, j);
						//updatePixel[j*config.resolutionX + i] = true;
					}
					else // Converges before maxIter
					{
						/*logzMag = abs(log(zMagSqr) / 2);
						//potential = log(logzMag / log(2)) / log(2);
						potential = log(logzMag) / log(2);
						dub = k + 1 - potential;			//this result is a double
						if (dub > maxIter || dub < -1000)
						dub = maxIter;
						escapeMatrix[j*config.resolutionX + i] = dub;
						writePixel(i, j);*/

						// Regular Shading
						//if (updatePixel[j*config.resolutionX + i] == true)
						{
							if (shortcut == false)
							{
								calculation.escapeBufferCPU[j*config.resolutionX + i] = k;
								writePixel(i, j);
							}
							//updatePixel[j*config.resolutionX + i] = false;
						}
					}
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

void single_avx_fractal_32()
{

}

void single_avx_fractal_64()
{

}

void single_fractal()
{

}

void fractalRender(int output)
{
	auto begin = std::chrono::high_resolution_clock::now();

	std::vector<std::thread> fractalT(config.threadMax);
	// Mode = 0 : Rectangular Subdivison
	if (config.executionMode == 0)
	{
		for (int sub = 0; sub < config.threadMax; ++sub)
		{
			if (output == 1)
			{
				printf("Thread %d initalized with parameters px:[%4d,%4d], py:[%4d,%4d]\n", sub,
					config.resolutionX / config.threadMax * sub, config.resolutionX / config.threadMax * (sub + 1),
					0, config.resolutionY
					);
			}
			switch (config.instructionSet)
			{
			case 0:
				fractalT[sub] = std::thread(avx_fractal_64, config.resolutionX / config.threadMax * sub, 0, config.resolutionX / config.threadMax * (sub + 1), config.resolutionY, sub, config.threadMax);
				break;
			case 1:
				fractalT[sub] = std::thread(avx_fractal, config.resolutionX / config.threadMax * sub, 0, config.resolutionX / config.threadMax * (sub + 1), config.resolutionY, sub, config.threadMax);
				break;
			case 2:
				fractalT[sub] = std::thread(fractal, config.resolutionX / config.threadMax * sub, 0, config.resolutionX / config.threadMax * (sub + 1), config.resolutionY, sub, config.threadMax);
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
						fractalT[thread] = std::thread(avx_fractal_64, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 1:
						fractalT[thread] = std::thread(avx_fractal, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 2:
						fractalT[thread] = std::thread(fractal, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					}
					if (output == 1)
					{
						printf("Thread %d initalized with parameters px:[%4d,%4d], py:[%4d,%4d]\n", thread,
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
						fractalT[thread] = std::thread(avx_fractal_64, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 1:
						fractalT[thread] = std::thread(avx_fractal, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 2:
						fractalT[thread] = std::thread(fractal, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					}
					if (output == 1)
					{
						printf("Thread %d initalized with parameters px:[%4d,%4d], py:[%4d,%4d]\n", thread,
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
						fractalT[thread] = std::thread(avx_fractal_64, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 1:
						fractalT[thread] = std::thread(avx_fractal, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					case 2:
						fractalT[thread] = std::thread(fractal, (config.resolutionX*x) / (config.threadMax / 2), (config.resolutionY*y) / (config.threadMax / 2), (config.resolutionX*(x + 1)) / (config.threadMax / 2) + (config.resolutionX - 2 * (config.resolutionX / (config.threadMax / 2))), (config.resolutionY*(y + 1)) / (config.threadMax / 2), thread, config.threadMax);
						break;
					}
					if (output == 1)
					{
						printf("Thread %d initalized with parameters px:[%4d,%4d], py:[%4d,%4d]\n", thread,
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

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	displayFunc();
	printf("Fractal Render Time %d\n",
		elasped
		);
}

////////////////////////////////////////
// Renderer
void actionZoom(double factor)
{
	double centerX = (config.xmax + config.xmin) / 2;
	double centerY = (config.ymax + config.ymin) / 2;
	config.zoomFactor = config.zoomFactor * factor;
	config.xmin = (centerX - ((double)config.resolutionX / config.resolutionY)*(config.zoomFactor));
	config.xmax = (centerX + ((double)config.resolutionX / config.resolutionY)*(config.zoomFactor));
	config.ymin = (centerY - (config.zoomFactor));
	config.ymax = (centerY + (config.zoomFactor));
	memset(calculation.updatePixel, true, config.resolutionX * config.resolutionY);
	calculation.screenStill = 0;
	fractalRender(0);
	printf("Window is x:[%9e, %9e] y:[%9e, %9e] r:[%9e] \n",
		config.xmin, config.xmax,
		config.ymin, config.ymax,
		config.zoomFactor
		);
}

void actionPoint(double centerX, double centerY, double factor)
{
	config.zoomFactor = factor;
	config.xmin = (centerX - ((double)config.resolutionX / config.resolutionY)*(config.zoomFactor));
	config.xmax = (centerX + ((double)config.resolutionX / config.resolutionY)*(config.zoomFactor));
	config.ymin = (centerY - (config.zoomFactor));
	config.ymax = (centerY + (config.zoomFactor));
	memset(calculation.updatePixel, true, config.resolutionX * config.resolutionY);
	calculation.screenStill = 0;
	fractalRender(0);
	printf("Window is x:[%9e, %9e] y:[%9e, %9e] r:[%9e] \n",
		config.xmin, config.xmax,
		config.ymin, config.ymax,
		config.zoomFactor
		);
}

void actionAdjustIterations(bool add_Mult, double factor)
{
	if (add_Mult == 0)
	{
		// Division
		if (factor < 1)
		{
			if (config.maxIter > 2)
			{
				calculation.prevMaxIter = config.maxIter;
				config.maxIter = config.maxIter * factor;
			}
		}
		// Multiply
		else
		{
			calculation.prevMaxIter = config.maxIter;
			config.maxIter = config.maxIter * factor;
		}
	}
	else
	{
		// Subtraction
		if (factor < 0)
		{
			if (config.maxIter > 2)
			{
				calculation.prevMaxIter = config.maxIter;
				config.maxIter = config.maxIter + factor;
			}
		}
		// Addition
		else
		{
			calculation.prevMaxIter = config.maxIter;
			config.maxIter = config.maxIter + factor;
		}
	}
	renderOptimizer(config.maxIter, config.maxIter - calculation.prevMaxIter);
	calculation.screenStill = 1;
	fractalRender(0);
	printf("Maximum iterations %d \n", config.maxIter);
}

void printError(int err, const char* msg)
{
	printf("Error: %d %s\n", err, msg);
}

void keyboard(GLFWwindow* window, int key, int scancode, int action, int modifier)
{
	// Action
	// GLFW_PRESS, GLFW_RELEASE, GLFW_REPEAT

	double centerX, centerY;
	if (action != GLFW_RELEASE)
	{
		// Regular Mode
		if (modifier == 0)
		{
			switch (key)
			{
			case 'W': /// Zoom in
				actionZoom(0.5);
				break;
			case 'A': /// Decrease iterations
				actionAdjustIterations(0, 0.5);
				break;
			case 'S': /// Zoom out
				actionZoom(2);
				break;
			case 'D': /// Increase iterations
				actionAdjustIterations(0, 2);
				break;
			case 'Q': /// Increase iterations auto
				for (int renderNumber = 0; renderNumber < 512; ++renderNumber)
				{
					calculation.prevMaxIter = config.maxIter;
					config.maxIter = config.maxIter + 1;
					renderOptimizer(config.maxIter, config.maxIter - calculation.prevMaxIter);
					calculation.screenStill = 1;
					fractalRender(0);
					printf("Maximum iterations %d \n", config.maxIter);
				}
				break;
			case 'E': /// Print center
				centerX = (config.xmax + config.xmin) / 2;
				centerY = (config.ymax + config.ymin) / 2;
				printf("Center (%.32f,%.32f) r[%.32f] \n", centerX, centerY, config.zoomFactor);
				break;
			case '/': /// Change CPU Instruction Set
				if (config.instructionSet == 0)
				{
					config.instructionSet = 1;
					printf("Instruction Set: AVX32\n");
				}
				else if (config.instructionSet == 1)
				{
					config.instructionSet = 2;
					printf("Instruction Set: Standard C\n");
				}
				else if (config.instructionSet == 2)
				{
					config.instructionSet = 0;
					printf("Instruction Set: AVX64\n");
				}
				break;
			case '.': /// Toggle Screen Optimization
				if (calculation.screenOptimization == 0)
				{
					calculation.screenOptimization = 1;
					printf("Pixel Optimization: On\n");
				}
				else
				{
					calculation.screenOptimization = 0;
					printf("Pixel Optimization: Off\n");
				}
				break;
			case GLFW_KEY_ESCAPE: /// Quit program
				glfwSetWindowShouldClose(window, GLFW_TRUE);
				break;
			default:
				break;
			}
		}
		// Coarse Adjustment
		if (modifier == GLFW_MOD_SHIFT)
		{

		}
		// Fine Adjustment
		else if (modifier == GLFW_MOD_CONTROL)
		{
			switch (key)
			{
			case 'W': /// Zoom in
				actionZoom(0.9);
				break;
			case 'A': /// Decrease iterations
				actionAdjustIterations(1, -1);
				break;
			case 'S': /// Zoom out
				actionZoom(1.1);
				break;
			case 'D': /// Increase iterations
				actionAdjustIterations(1, 1);
				break;
			default:
				break;
			}

		}
	}
}

void mouseClick(GLFWwindow* window, int button, int action, int mods)
{
	double mouseX, mouseY;
	glfwGetCursorPos(window, &mouseX, &mouseY);

	double centerX, centerY;
	double windowX, windowY;
	mouseY = config.resolutionY - mouseY; // OpenGL textures start at bottom left corner
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)	// Left Mouse Button
	{
			centerX = (config.xmax + config.xmin) / 2;
			centerY = (config.ymax + config.ymin) / 2;
			windowX = (config.xmax - config.xmin) / 2;
			windowY = (config.ymax - config.ymin) / 2;
			//std::cout << "MouseUp on " << mouseX << " " << mouseY << std::endl;
			config.xmin = config.xmin + (windowX*((double)(mouseX - config.resolutionX / 2) * 2 / config.resolutionX));
			config.xmax = config.xmax + (windowX*((double)(mouseX - config.resolutionX / 2) * 2 / config.resolutionX));
			config.ymin = config.ymin + (windowY*(-1)*((double)(mouseY - config.resolutionY / 2) * 2 / config.resolutionY));
			config.ymax = config.ymax + (windowY*(-1)*((double)(mouseY - config.resolutionY / 2) * 2 / config.resolutionY));
			memset(calculation.updatePixel, true, config.resolutionX * config.resolutionY);
			calculation.screenStill = 0;
			fractalRender(1);
			printf("Window is x:[%9e, %9e] y:[%9e, %9e] r:[%9e] \n",
				config.xmin, config.xmax,
				config.ymin, config.ymax,
				config.zoomFactor
				);
	}
}

void mouseMotion(GLFWwindow* window, double x, double y)
{
	int mouseX = (int)x;
	int mouseY = (int)y;
	double centerX = (config.xmax + config.xmin) / 2;
	double centerY = (config.ymax + config.ymin) / 2;
	double windowX = (config.xmax - config.xmin) / 2;
	double windowY = (config.ymax - config.ymin) / 2;

	printf("(px, py) %4d %4d (zi, zr) %9f %9f (i) %4d (r, g, b) %3d %3d %3d Update> %d\n",
		mouseX, mouseY,
		centerX + windowX*((double)(mouseX - config.resolutionX / 2) * 2 / config.resolutionX), centerY + windowY*(-1)*((double)(mouseY - config.resolutionY / 2) * 2 / config.resolutionY),
		(int)calculation.escapeBufferCPU[config.resolutionX*mouseY + mouseX],
		(int)calculation.screenBufferCPU[(4 * (config.resolutionX*mouseY + mouseX) + 0)], (int)calculation.screenBufferCPU[(4 * (config.resolutionX*mouseY + mouseX) + 1)], (int)calculation.screenBufferCPU[(4 * (config.resolutionX*mouseY + mouseX) + 2)],
		(int)calculation.updatePixel[config.resolutionX*mouseY + mouseX]
		);

	//double fx = (double)(x - lastx) / 50.0 / (double)(imageW);
	//double fy = (double)(lasty - y) / 50.0 / (double)(imageH);
}

void displayFunc()
{
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, config.resolutionX, config.resolutionY, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)calculation.screenBufferCPU);

	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glDisable(GL_DEPTH_TEST);

	glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	glBegin(GL_QUADS);

	// All verticies are flipped to start at upper left corner
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(0.0f, 1.0f);

	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, 1.0f);

	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 0.0f);

	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(0.0f, 0.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);

	glfwSwapBuffers(window);
}

GLuint compileASMShader(GLenum program_type, const char *code)
{
	GLuint program_id;
	glGenProgramsARB(1, &program_id);
	glBindProgramARB(program_type, program_id);
	glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte *)code);

	GLint error_pos;
	glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

	if (error_pos != -1)
	{
		const GLubyte *error_string;
		error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
		fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
		return 0;
	}

	return program_id;
}

void deleteBuffers()
{
	if (calculation.screenBufferCPU)
	{
		delete[] calculation.screenBufferCPU;
		calculation.screenBufferCPU = NULL;
	}

	if (calculation.escapeBufferCPU)
	{
		delete[] calculation.escapeBufferCPU;
		calculation.escapeBufferCPU = NULL;
	}

	if (calculation.updatePixel)
	{
		delete[] calculation.updatePixel;
		calculation.updatePixel = NULL;
	}

	if (gl_Tex)
	{
		glDeleteTextures(1, &gl_Tex);
		gl_Tex = NULL;
	}

	if (gl_PBO)
	{
		glDeleteBuffers(1, &gl_PBO);
		gl_PBO = NULL;
	}
}

bool initializeBuffers(int width, int height)
{
	// Flush buffers
	deleteBuffers();

	// Check for minimized window
	if ((width == 0) && (height == 0))
	{
		return false;
	}

	// Allocate Buffers
	std::cout << "Resolution Set to " << width << " " << height << std::endl;
	calculation.escapeBufferCPU = new double[width * height];
	calculation.screenBufferCPU = new unsigned char[width * height * 4];
	calculation.updatePixel = new bool[width * height];
	memset(calculation.updatePixel, true, width * height);

	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*)calculation.screenBufferCPU);
	glEnable(GL_TEXTURE_2D);

	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, calculation.screenBufferCPU, GL_STREAM_COPY);

	gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, "!!ARBfp1.0\n"
		"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
		"END");

	displayFunc();
	displayFunc();
	return true;
}

void resize(GLFWwindow* window, int width, int height)
{
	initializeBuffers(width, height);
	config.resolutionX = width;
	config.resolutionY = height;

	glViewport(0, 0, width, height);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

	displayFunc();
	displayFunc();
	fractalRender(0);
}

void initializeGL()
{
	glfwSetErrorCallback(printError);
	if (!glfwInit())
	{
		exit(EXIT_FAILURE);
	}
	window = glfwCreateWindow(config.resolutionX, config.resolutionY, "FractalGL", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	// Set GLFW callback functions
	glfwSetKeyCallback(window, keyboard);
	glfwSetMouseButtonCallback(window, mouseClick);
	glfwSetCursorPosCallback(window, mouseMotion);
	glfwSetWindowSizeCallback(window, resize);


	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	glewExperimental = GL_TRUE;
	glewInit();

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glViewport(0, 0, config.resolutionX, config.resolutionY);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);


	const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
	const GLubyte* version = glGetString(GL_VERSION); // version as a string
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);
}

void initalizeConfig()
{
	//Load configuration
	std::ifstream options;
	std::string line;
	std::string token;
	std::string field;

	options.open("options.cfg");
	while (!options.eof())
	{
		std::getline(options, line);
		token = line.substr(0, line.find(' '));
		field = line.substr(line.find(' ') + 1, line.find(';'));
		if (token.compare("Iterations") == 0)
		{
			config.maxIter = std::stoi(field);
		}
		if (token.compare("Colors_Used") == 0)
		{
			config.colorsUsed = std::stoi(field);
		}
		if (token.compare("Color_Density") == 0)
		{
			config.colorDensity = std::stoi(field);
		}
		if (token.compare("Xcoord") == 0)
		{
			config.xcoord = std::stoi(field);
		}
		if (token.compare("Ycoord") == 0)
		{
			config.ycoord = std::stoi(field);
		}
		if (token.compare("Rzoom") == 0)
		{
			config.zoomFactor = std::stoi(field);
		}
		if (token.compare("InstructionSet") == 0)
		{
			config.instructionSet = std::stoi(field);
		}
		if (token.compare("Threads") == 0)
		{
			config.threadMax = std::stoi(field);
		}
		if (token.compare("ShadingMode") == 0)
		{
			config.shadingMode = std::stoi(field);
		}
		if (token.compare("ExecutionMode") == 0)
		{
			config.executionMode = std::stoi(field);
		}
		if (token.compare("PaletteFile") == 0)
		{
			config.paletteFile = field;
		}
		if (token.compare("//") == 0)
		{
			std::getline(options, line);	// Skip line
		}
		if (token.compare("/n") == 0)
		{
			std::getline(options, line);	// Skip line
		}
	}
	options.close();

	//Load palette
	std::ifstream palette;
	palette.open(config.paletteFile);
	int paletteRed, paletteGreen, paletteBlue;

	std::getline(palette, line);
	token = line.substr(0, line.find(' '));
	field = line.substr(line.find(' ') + 1, line.find(';'));
	if (token.compare("Colors") == 0)
	{
		config.paletteNumber = std::stoi(field);
		std::cout << "Colors " << config.paletteNumber << std::endl;
	}

	config.palette = new unsigned char[(config.paletteNumber + 1) * 4];

	for (int paletteIndex = 0; paletteIndex < config.paletteNumber; paletteIndex++)
	{
		std::getline(palette, line);
		std::stringstream stream(line);
		stream >> paletteRed;
		stream.ignore();
		stream >> paletteGreen;
		stream.ignore();
		stream >> paletteBlue;
		stream.ignore();

		config.palette[4 * paletteIndex + 0] = (unsigned char)paletteRed;
		config.palette[4 * paletteIndex + 1] = (unsigned char)paletteGreen;
		config.palette[4 * paletteIndex + 2] = (unsigned char)paletteBlue;
		config.palette[4 * paletteIndex + 3] = 255;

		std::cout << "Palette " << paletteIndex << ": " << (int)config.palette[4 * paletteIndex + 0] << " " << (int)config.palette[4 * paletteIndex + 1] << " " << (int)config.palette[4 * paletteIndex + 2] << std::endl;
	}
	palette.close();

	//modify the palette so that colors do not immediately switch at the boundary
	config.palette[4 * (config.paletteNumber) + 0] = config.palette[0];
	config.palette[4 * (config.paletteNumber) + 1] = config.palette[1];
	config.palette[4 * (config.paletteNumber) + 2] = config.palette[2];
	config.palette[4 * (config.paletteNumber) + 3] = config.palette[3];

	config.xmin = (config.xcoord - ((double)config.resolutionX / config.resolutionY)*(config.zoomFactor));
	config.xmax = (config.xcoord + ((double)config.resolutionX / config.resolutionY)*(config.zoomFactor));
	config.ymin = (config.ycoord - (config.zoomFactor));
	config.ymax = (config.ycoord + (config.zoomFactor));
}

void renderLoop()
{
	initializeGL();
	initializeBuffers(config.resolutionX, config.resolutionY);
	do // Main Loop
	{
		glfwPollEvents();
		//pollCommandPrompt();
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	} while (!glfwWindowShouldClose(window));

	glfwDestroyWindow(window);
	glfwTerminate();
	deleteBuffers();
}

////////////////////////////////////////
// Command Handling
void pollCommandPrompt()
{

}

void display_help()
{
	printf("\nOpenGL Renderer Commands\n");
	printf("[w] Zoom in : 2x\n");
	printf("[s] Zoom out : 2x\n");
	printf("[a] Increase iterations : 2x\n");
	printf("[d] Decrease iterations : 2x\n\n");

	printf("SHIFT + [w] Zoom in : 1.1x\n");
	printf("SHIFT + [s] Zoom out : 1.1x\n");
	printf("SHIFT + [a] Increase iterations : 1\n");
	printf("SHIFT + [d] Decrease iterations : 1\n\n");

	printf("[q] Auto iterate : 256\n");
	printf("[/] Change Instruction Set\n");
	printf("[.] Toggle Render Optimization\n\n");

	printf("\nTerminal Mode Commands\n");
	printf("\"point x y r\" \t refocus the graph onto a specified point.\n");
	printf("\"resize x y\" \t resize the resolution of the window.\n");
}

void command_resize(int width, int height)
{
	if (window)
	{
		resize(window, width, height);
	}
	else
	{
		printf("Graph not running!");
	}
}

void command_point(int x, int y, int r)
{
	if (window)
	{

	}
	else
	{
		printf("Graph not running!");
	}
}

int main(int argc, char **argv)
{
	initalizeConfig();

	std::thread renderWindow;
	renderWindow = std::thread(renderLoop);

	std::ifstream options;
	std::string line;
	std::string command;
	std::string field;
	do
	{
		std::getline(std::cin, line);
		command = line.substr(0, line.find(' '));
		field = line.substr(line.find(' ') + 1, line.find('\n'));
		if (command.compare("?") == 0 || command.compare("help") == 0)
		{
			display_help();
		}
		else if (command.compare("point") == 0)
		{
			int x, y, r;

			std::stringstream stream(field);
			stream >> x;
			stream.ignore();
			stream >> y;
			stream.ignore();
			stream >> r;
			stream.ignore();

			actionPoint(x, y, r);
		}
		else if (command.compare("resize") == 0)
		{
			int x, y;

			std::stringstream stream(field);
			stream >> x;
			stream.ignore();
			stream >> y;
			stream.ignore();

			glfwSetWindowSize(window, x, y);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	} while (true);

	renderWindow.join();
	exit(EXIT_SUCCESS);

	return 0;
}
////////////////////////////////////////
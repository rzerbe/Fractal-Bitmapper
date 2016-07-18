#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>
#include <stdint.h>
#include <string>
#include <fstream>
#include <ostream>

#include <chrono>
#include <thread>
#include <immintrin.h>
#include <intrin.h>

#include <SDL.h>
/*
ctrl-F "objective" to find potential things to work on, in no particular order.

objective1:
Redesign our fractal() function to be called each time wedo the computation for a new image.
Take some of what we have in the main() function and design a
render() function that converts the escapeMatrix to a colored pixel array.
Easily generate multiple images with a single run of this program (useful for getting various magnifications).

objective2: (complete)
implement CPU multithreading

objective3:
implement GPU acceleration

objective4:
implement arbitrary precision computation with some high precision math library

objective5:
generate two images.
one image with coordinate axes to orient our position within the fractal.
the other image strictly for aesthetic.

objective6:
display intermediate images with reduced resolution during computation, and gradually improve resolution
in real time as computation continues (will have to adjust the order in which we compute the pixels).

*/

//apparently 30000 is too much
const int w = 1000;
const int h = 1000;

const int upperBound = 10000;
int maxIter = 256;
const int numColors = 8;
int colorDensity;
int colorsUsed;
double zoomFactor = 1;
int shadingMode = 0;
int threadMax = 4;
int instructionSet = 0;
int executionMode = 0;

void writePixel(int x, int y);

double xmin, xmax, ymin, ymax;
double xcoord, ycoord, windowWidth, windowHeight;
float escapeMatrix[w*h];
unsigned char screenBuffer[3 * w * h];
void adjustParameters()
{
	std::string choice;
	std::cout << "Enter a point to be centered on, or a window? point/standardpt/window\n";
	std::cin >> choice;

	if (choice == "standardpt")
	{
		std::cout << "Enter x coord:\n";
		std::cin >> xcoord;
		std::cout << "Enter y coord:\n";
		std::cin >> ycoord;
		std::cout << "Zoom Factor:\n";
		std::cin >> zoomFactor;

		xmin = (xcoord - ((double)w/h)*(zoomFactor));
		xmax = (xcoord + ((double)w/h)*(zoomFactor));
		ymin = (ycoord - (zoomFactor));
		ymax = (ycoord + (zoomFactor));
	}

	if (choice == "point")
	{
		std::cout << "Enter x coord:\n";
		std::cin >> xcoord;
		std::cout << "Enter y coord:\n";
		std::cin >> ycoord;
		std::cout << "Enter window width:\n";
		std::cin >> windowWidth;
		std::cout << "Enter window height:\n";
		std::cin >> windowHeight;

		xmin = xcoord - windowWidth / 2;
		xmax = xcoord + windowWidth / 2;
		ymin = ycoord - windowHeight / 2;
		ymax = ycoord + windowHeight / 2;

		/*
		The below input variabes are not used yet.
		It is not trivial to implement because of the way our code is set up.
		
		std::cout << "Do you wish to generate multiple images? y/n\n";
		std::cin >> multiple;
		std::cout << "Enter zoom factor:\n";
		std::cin >> zoomFactor;
		std::cout << "Enter number of images to generate:\n";
		std::cin >> numImages;
		*/
	}
	if (choice == "window")
	{
		std::cout << "Enter xmin:\n";
		std::cin >> xmin;
		std::cout << "Enter xmax:\n";
		std::cin >> xmax;
		std::cout << "Enter ymin:\n";
		std::cin >> ymin;
		std::cout << "Enter ymax:\n";
		std::cin >> ymax;
	}
}

void avx_fractal(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads)
{
	auto begin = std::chrono::high_resolution_clock::now();

	float constants[] = { (xmax-xmin)/w, (ymax-ymin)/h, xmin, ymin, 1.0f, 4.0f, j_init};
	__m256 ymm0 = _mm256_broadcast_ss(constants);   // all dx
	__m256 ymm1 = _mm256_broadcast_ss(constants + 1); // all dy
	__m256 ymm2 = _mm256_broadcast_ss(constants + 2); // all x1
	__m256 ymm3 = _mm256_broadcast_ss(constants + 3); // all y1
	__m256 ymm4 = _mm256_broadcast_ss(constants + 4); // all 1's (iter increments)
	__m256 ymm5 = _mm256_broadcast_ss(constants + 5); // all 4's (comparisons)
	__m256 ymm6 = _mm256_broadcast_ss(constants + 6); // set to starting y position

	float incr[8] = { 0.0f + i_init,1.0f + i_init,2.0f + i_init,3.0f + i_init,4.0f + i_init,5.0f + i_init,6.0f + i_init, 7.0f + i_init }; // used to reset the i position when j increases

	for (int j = j_init; j < j_fin; j += 1)
	{
		__m256 ymm7 = _mm256_set_ps(incr[0], incr[1], incr[2], incr[3], incr[4], incr[5], incr[6], incr[7]);  // i counter set to 0,1,2,..,7
		for (int i = i_init; i < i_fin; i += 8)
		{
			__m256 ymm8 = _mm256_mul_ps(ymm7, ymm0);  // x0 = (i+k)*dx 
			ymm8 = _mm256_add_ps(ymm8, ymm2);         // x0 = x1+(i+k)*dx
			__m256 ymm9 = _mm256_mul_ps(ymm6, ymm1);  // y0 = j*dy
			ymm9 = _mm256_add_ps(ymm9, ymm3);         // y0 = y1+j*dy
			__m256 ymm10 = _mm256_xor_ps(ymm0, ymm0);  // zero out iteration counter
			__m256 ymm11 = ymm10, ymm12 = ymm10;        // set initial xi=0, yi=0

			int test = 0;
			unsigned int iter = 0;
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
			} while ((test != 0) && (iter < maxIter));


			// write only where needed
			int top = (i + 7) < w ? 8 : w & 7;
			for (int k = 0; k < top; ++k)
			{
				escapeMatrix[i + k + j*w] = ymm10.m256_f32[top - k - 1];
				writePixel(i + k, j);
			}

			// next i position - increment each slot by 8
			ymm7 = _mm256_add_ps(ymm7, ymm5);
			ymm7 = _mm256_add_ps(ymm7, ymm5);
		}
		ymm6 = _mm256_add_ps(ymm6, ymm4); // increment j counter
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "AVX Thread " << threadID << "/" << maxThreads << " finished in " << elasped << " milliseconds " << std::endl;
}

void fractal(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads)
{
	auto begin = std::chrono::high_resolution_clock::now();
	double zrCurrent;
	double ziCurrent;
	double zrPrevious;
	double ziPrevious;

	double xWindow = xmax - xmin;
	double yWindow = ymax - ymin;

	int m;
	double dub;
	int k;
	double ci, cr, q, logzMag, zMagSqr, potential;
	bool shortcut;

	for (int j = j_init; j < j_fin; j++) // y axis
	{
		/*//give percentage progress
		m = j % (h / 100);
		if (m == 0)
		{
			std::cout << "Thread: " << threadID << " Computation " << (int)(((double)(j - j_init) / (j_fin - j_init) * 100)) << "% complete.\n";
		}*/
		ci = ymin + ((double)j) * (yWindow) / h;
		
		for (int i = i_init; i < i_fin; i++) // x axis
		{
			shortcut = false;
			cr = xmin + ((double)i) * (xWindow) / w;

			//check whether point lies in main cardioid
			/*q = (cr - 0.25)*(cr - 0.25) + ci*ci;
			if (q*(q + cr - 0.25) - 0.25*ci*ci < 0)
			{
				escapeMatrix[j*w + i] = maxIter;
				writePixel(i, j);
				shortcut = true;
			}
			//check whether point lies in period-2 bulb
			else if ((cr + 1)*(cr + 1) + ci*ci - 0.0625 < 0)
			{
				escapeMatrix[j*w + i] = maxIter;
				writePixel(i, j);
				shortcut = true;
			}*/

			//if point lies within main cardioid or bulb, skip computation
			if (!shortcut)
			{
				zrPrevious = 0;
				ziPrevious = 0;

				for (k = 1; k < maxIter; k++)
				{
					//Optimized
					zrCurrent = zrPrevious * zrPrevious - ziPrevious * ziPrevious + cr;
					ziCurrent = 2 * zrPrevious * ziPrevious + ci;
					zMagSqr = zrCurrent * zrCurrent + ziCurrent * ziCurrent;

					zrPrevious = zrCurrent;
					ziPrevious = ziCurrent;

					if (zMagSqr > 66536)		//did some optimizations in this area
					{
						break;
					}
				}
				//Non-convergence (k was stored as -2^31 - 1 if not handled)
				if (k == maxIter)
				{
					escapeMatrix[j*w + i] = maxIter;
					writePixel(i, j);
				}
				else // Converges before maxIter
				{
					/*logzMag = abs(log(zMagSqr) / 2);
					//potential = log(logzMag / log(2)) / log(2);
					potential = log(logzMag) / log(2);
					dub = k + 1 - potential;			//this result is a double
					if (dub > maxIter || dub < -1000)
						dub = maxIter;
					escapeMatrix[j*w + i] = dub;
					writePixel(i, j);*/

					// Regular Shading
					escapeMatrix[j*w + i] = k;
					writePixel(i, j);
				}
			}
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "Standard Thread " << threadID << "/" << maxThreads << " finished in " << elasped << " milliseconds " << std::endl;
}

void writePixel(int x, int y)
{
	//colorDraw is the color we will draw with
	//color1 and color2 are the colors that need to be linearly interpolated to get colorDraw
	//palette stores the rgb of all of our hardcoded colors

	if (escapeMatrix[y*h + x] == maxIter)
	{
		screenBuffer[3 * (y*h + x) + 0] = 0;
		screenBuffer[3 * (y*h + x) + 1] = 0;
		screenBuffer[3 * (y*h + x) + 2] = 0;
	}
	else
	{
		if (shadingMode == 0)
		{
			unsigned char color1[3];
			unsigned char color2[3];
			unsigned char palette[numColors + 1][3];

			//orange
			palette[0][0] = 255;
			palette[0][1] = 128;
			palette[0][2] = 0;

			//green
			palette[1][0] = 0;
			palette[1][1] = 255;
			palette[1][2] = 0;

			//violet
			palette[2][0] = 255;
			palette[2][1] = 0;
			palette[2][2] = 255;

			//blue
			palette[3][0] = 0;
			palette[3][1] = 0;
			palette[3][2] = 255;

			//red
			palette[4][0] = 255;
			palette[4][1] = 0;
			palette[4][2] = 0;

			//yellow
			palette[5][0] = 255;
			palette[5][1] = 255;
			palette[5][2] = 0;

			//purple
			palette[6][0] = 153;
			palette[6][1] = 0;
			palette[6][2] = 153;

			//white
			palette[7][0] = 255;
			palette[7][1] = 255;
			palette[7][2] = 255;

			//red
			palette[8][0] = 255;
			palette[8][1] = 0;
			palette[8][2] = 0;

			double k = escapeMatrix[y*h + x];

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


			//modify the palette so that colors do not immediately switch at the boundary
			palette[colorsUsed][0] = palette[0][0];
			palette[colorsUsed][1] = palette[0][1];
			palette[colorsUsed][2] = palette[0][2];

			//colorDraw is the color we will draw with
			//color1 and color2 are the colors that need to be linearly interpolated to get colorDraw

			//figure out which two colors our point should be interpolated between, based on k
			double kscaled = fmod(k, colorDensity) / colorDensity * (colorsUsed);
			color1[0] = palette[(int)floor(kscaled)][0];
			color1[1] = palette[(int)floor(kscaled)][1];
			color1[2] = palette[(int)floor(kscaled)][2];
			color2[0] = palette[(int)floor(kscaled + 1)][0];
			color2[1] = palette[(int)floor(kscaled + 1)][1];
			color2[2] = palette[(int)floor(kscaled + 1)][2];

			//linearly interpolate between color1 and color2, based on fractional part of k
			screenBuffer[3 * (y*h + x) + 0] = (color2[0] - color1[0])*(kscaled - floor(kscaled)) + color1[0];
			screenBuffer[3 * (y*h + x) + 1] = (color2[1] - color1[1])*(kscaled - floor(kscaled)) + color1[1];
			screenBuffer[3 * (y*h + x) + 2] = (color2[2] - color1[2])*(kscaled - floor(kscaled)) + color1[2];
		}
		else if (shadingMode == 1)
		{
			//Monochromatic
			screenBuffer[3 * (y*h + x) + 0] = maxIter - escapeMatrix[y*h + x];
			screenBuffer[3 * (y*h + x) + 1] = maxIter - escapeMatrix[y*h + x];
			screenBuffer[3 * (y*h + x) + 2] = maxIter - escapeMatrix[y*h + x];
		}
	}

}

void writeBitmap(int i_init, int j_init, int i_fin, int j_fin)
{
	//Bitmap Header
	unsigned int filesize = 54 + 3 * w*h;  //w is your image width, h is image height, both int
										   /*if (img)
										   free(img);
										   img = (unsigned char *)malloc(3 * w*h);
										   memset(img, 0, sizeof(img));*/

	unsigned char bmpfileheader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
	unsigned char bmpinfoheader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0 };

	bmpfileheader[2] = (unsigned char)(filesize);
	bmpfileheader[3] = (unsigned char)(filesize >> 8);
	bmpfileheader[4] = (unsigned char)(filesize >> 16);
	bmpfileheader[5] = (unsigned char)(filesize >> 24);

	bmpinfoheader[4] = (unsigned char)(w);
	bmpinfoheader[5] = (unsigned char)(w >> 8);
	bmpinfoheader[6] = (unsigned char)(w >> 16);
	bmpinfoheader[7] = (unsigned char)(w >> 24);
	bmpinfoheader[8] = (unsigned char)(h);
	bmpinfoheader[9] = (unsigned char)(h >> 8);
	bmpinfoheader[10] = (unsigned char)(h >> 16);
	bmpinfoheader[11] = (unsigned char)(h >> 24);

	FILE *bitmap = fopen("img1.bmp", "wb");
	fwrite(bmpfileheader, 1, 14, bitmap);
	fwrite(bmpinfoheader, 1, 40, bitmap);

	//colorDraw is the color we will draw with
	//color1 and color2 are the colors that need to be linearly interpolated to get colorDraw
	//palette stores the rgb of all of our hardcoded colors
	unsigned char color1[3];
	unsigned char color2[3];
	unsigned char palette[numColors+1][3];
	unsigned char colorDraw[3 * w];
	unsigned char bmppad[3] = { 0,0,0 };

	//orange
	palette[0][0] = 255;
	palette[0][1] = 128;
	palette[0][2] = 0;

	//green
	palette[1][0] = 0;
	palette[1][1] = 255;
	palette[1][2] = 0;

	//violet
	palette[2][0] = 255;
	palette[2][1] = 0;
	palette[2][2] = 255;

	//blue
	palette[3][0] = 0;
	palette[3][1] = 0;
	palette[3][2] = 255;

	//red
	palette[4][0] = 255;
	palette[4][1] = 0;
	palette[4][2] = 0;

	//yellow
	palette[5][0] = 255;
	palette[5][1] = 255;
	palette[5][2] = 0;

	//purple
	palette[6][0] = 153;
	palette[6][1] = 0;
	palette[6][2] = 153;

	//white
	palette[7][0] = 255;
	palette[7][1] = 255;
	palette[7][2] = 255;

	//red
	palette[8][0] = 255;
	palette[8][1] = 0;
	palette[8][2] = 0;


	for (int j = 0; j < h; j++) // y axis
	{
		//give percentage progress
		/*int m = j % (h / 100);
		if (m == 0)
			std::cout << "Coloring " << (int)(((double)(j) / h * 100)) << "% complete.\n";*/

		for (int i = 0; i < w; i++) // x axis
		{

			if (escapeMatrix[j*w + i] == maxIter)
			{
				colorDraw[3*i + 0] = 0;
				colorDraw[3*i + 1] = 0;
				colorDraw[3*i + 2] = 0;
			}
			else
			{
				double k = escapeMatrix[j*w + i];

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


				//modify the palette so that colors do not immediately switch at the boundary
				palette[colorsUsed][0] = palette[0][0];
				palette[colorsUsed][1] = palette[0][1];
				palette[colorsUsed][2] = palette[0][2];

				//colorDraw is the color we will draw with
				//color1 and color2 are the colors that need to be linearly interpolated to get colorDraw

				//figure out which two colors our point should be interpolated between, based on k
				double kscaled = fmod(k, colorDensity) / colorDensity * (colorsUsed);
				color1[0] = palette[(int)floor(kscaled)][0];
				color1[1] = palette[(int)floor(kscaled)][1];
				color1[2] = palette[(int)floor(kscaled)][2];
				color2[0] = palette[(int)floor(kscaled + 1)][0];
				color2[1] = palette[(int)floor(kscaled + 1)][1];
				color2[2] = palette[(int)floor(kscaled + 1)][2];

				//linearly interpolate between color1 and color2, based on fractional part of k
				colorDraw[3*i + 0] = (color2[0] - color1[0])*(kscaled - floor(kscaled)) + color1[0];
				colorDraw[3*i + 1] = (color2[1] - color1[1])*(kscaled - floor(kscaled)) + color1[1];
				colorDraw[3*i + 2] = (color2[2] - color1[2])*(kscaled - floor(kscaled)) + color1[2];

				//Monochromatic
				/*r[0] = 255 - k;
				g[0] = 255 - k;
				b[0] = 255 - k;*/
			}
		}
		fwrite(&colorDraw, 1, 3 * w, bitmap);
		fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, bitmap);
	}
	fclose(bitmap);
}

void fractalRender(SDL_Renderer *renderer, SDL_Rect rect, int output)
{
	//CPU Multithreading
	auto begin = std::chrono::high_resolution_clock::now();

	std::vector<std::thread> fractalT(threadMax);

	// Mode = 0 : Rectangular Subdivison
	if (executionMode == 0)
	{
		for (int sub = 0; sub < threadMax; ++sub)
		{
			if (sub == threadMax - 1)
			{
				if (output == 1)
				{
					std::cout << "Thread Main " << sub << " initialized with parameters"
						<< " xinit " << w / threadMax * sub
						<< " yinit " << 0
						<< " xfin " << w / threadMax * (sub + 1)
						<< " yfin " << h << std::endl;
				}
				switch (instructionSet)
				{
				case 0:
					fractal(w / threadMax * sub, 0, w / threadMax * (sub + 1), h, threadMax - 1, threadMax);
					break;
				case 1:
					avx_fractal(w / threadMax * sub, 0, w / threadMax * (sub + 1), h, threadMax - 1, threadMax);
					break;
				}
			}
			else
			{
				if (output == 1)
				{
					std::cout << "Thread " << sub << " initialized with parameters"
						<< " xinit " << w / threadMax * sub
						<< " yinit " << 0
						<< " xfin " << w / threadMax * (sub + 1)
						<< " yfin " << h << std::endl;
				}
				switch (instructionSet)
				{
				case 0:
					fractalT[sub] = std::thread(fractal, w / threadMax * sub, 0, w / threadMax * (sub + 1), h, sub, threadMax);
					break;
				case 1:
					fractalT[sub] = std::thread(avx_fractal, w / threadMax * sub, 0, w / threadMax * (sub + 1), h, sub, threadMax);
					break;
				}
			}
		}
	}
	else if (executionMode == 1)
	{
		int thread = 0;
		for (int y = 0; y < threadMax / 2; ++y) // Starting y subdivison
		{
			for (int x = 0; x < (threadMax / 2); ++x) // Starting x subdivison
			{
				if (thread == threadMax - 1) // Allow main thread to perform computation
				{
					if (output == 1)
					{
						std::cout << "Thread Main " << thread << " initialized with parameters"
							<< " xinit " << (w*x) / (threadMax / 2)
							<< " yinit " << (h*y) / (threadMax / 2)
							<< " xfin " << (w*(x + 1)) / (threadMax / 2)
							<< " yfin " << (h*(y + 1)) / (threadMax / 2) << std::endl;
					}
					switch (instructionSet)
					{
					case 0:
						fractal(w / (threadMax / 2), h / (threadMax / 2), w, h, threadMax - 1, threadMax);
						break;
					case 1:
						avx_fractal(w / (threadMax / 2), h / (threadMax / 2), w, h, threadMax - 1, threadMax);
						break;
					}
				}
				else // Create new thread
				{
					if (x == threadMax / 2)
					{
						switch (instructionSet)
						{
						case 0:
							fractalT[thread] = std::thread(fractal, (w*x) / (threadMax / 2), (h*y) / (threadMax / 2), (w*(x + 1)) / (threadMax / 2) + (w - 2 * (w / (threadMax / 2))), (h*(y + 1)) / (threadMax / 2), thread, threadMax);
							break;
						case 1:
							fractalT[thread] = std::thread(avx_fractal, (w*x) / (threadMax / 2), (h*y) / (threadMax / 2), (w*(x + 1)) / (threadMax / 2) + (w - 2 * (w / (threadMax / 2))), (h*(y + 1)) / (threadMax / 2), thread, threadMax);
							break;
						}
						if (output == 1)
						{
							std::cout << "Thread " << thread << " initialized with parameters"
								<< " xinit " << (w*x) / (threadMax / 2)
								<< " yinit " << (h*y) / (threadMax / 2)
								<< " xfin " << (w*(x + 1)) / (threadMax / 2) + (w - 2 * (w / (threadMax / 2)))
								<< " yfin " << (h*(y + 1)) / (threadMax / 2) << std::endl;
						}
						thread++;
					}
					else if (y == threadMax / 2)
					{
						switch (instructionSet)
						{
						case 0:
							fractalT[thread] = std::thread(fractal, (w*x) / (threadMax / 2), (h*y) / (threadMax / 2), (w*(x + 1)) / (threadMax / 2) + (w - 2 * (w / (threadMax / 2))), (h*(y + 1)) / (threadMax / 2), thread, threadMax);
							break;
						case 1:
							fractalT[thread] = std::thread(avx_fractal, (w*x) / (threadMax / 2), (h*y) / (threadMax / 2), (w*(x + 1)) / (threadMax / 2) + (w - 2 * (w / (threadMax / 2))), (h*(y + 1)) / (threadMax / 2), thread, threadMax);
							break;
						}
						if (output == 1)
						{
							std::cout << "Thread " << thread << " initialized with parameters"
								<< " xinit " << (w*x) / (threadMax / 2)
								<< " yinit " << (h*y) / (threadMax / 2)
								<< " xfin " << (w*(x + 1)) / (threadMax / 2) + (w - 2 * (w / (threadMax / 2)))
								<< " yfin " << (h*(y + 1)) / (threadMax / 2) << std::endl;
						}
						thread++;
					}
					else
					{
						switch (instructionSet)
						{
						case 0:
							fractalT[thread] = std::thread(fractal, (w*x) / (threadMax / 2), (h*y) / (threadMax / 2), (w*(x + 1)) / (threadMax / 2) + (w - 2 * (w / (threadMax / 2))), (h*(y + 1)) / (threadMax / 2), thread, threadMax);
							break;
						case 1:
							fractalT[thread] = std::thread(avx_fractal, (w*x) / (threadMax / 2), (h*y) / (threadMax / 2), (w*(x + 1)) / (threadMax / 2) + (w - 2 * (w / (threadMax / 2))), (h*(y + 1)) / (threadMax / 2), thread, threadMax);
							break;
						}
						if (output == 1)
						{
							std::cout << "Thread " << thread << " initialized with parameters"
								<< " xinit " << (w*x) / (threadMax / 2)
								<< " yinit " << (h*y) / (threadMax / 2)
								<< " xfin " << (w*(x + 1)) / (threadMax / 2) + (w - 2 * (w / (threadMax / 2)))
								<< " yfin " << (h*(y + 1)) / (threadMax / 2) << std::endl;
						}
						thread++;
					}
				}
			}
		}
	}

	for (int i = 0; i < threadMax - 1; ++i)
	{
		fractalT[i].join();
	}

	SDL_Surface *surface_window = SDL_CreateRGBSurfaceFrom(&screenBuffer, w, h, 3 * 8, w * 3, 0xFF0000, 0x00FF00, 0x0000FF, 0); // <- cannot do rectangular regions?
	if (surface_window == NULL)
	{
		std::cout << "SDL_CreateRGBSurfaceFrom Error: " << SDL_GetError() << std::endl;
	}
	SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface_window);
	SDL_FreeSurface(surface_window);
	SDL_RenderCopyEx(renderer, texture, NULL, &rect, 0, NULL, SDL_FLIP_VERTICAL);
	SDL_DestroyTexture(texture);
	SDL_RenderPresent(renderer);

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << "Fractal Render Time " << elasped << std::endl;
}

int main(int argc, char** argv)
{
	//escapeMatrix = (short*)malloc(w * h * sizeof(short));

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
			maxIter = std::stoi(field);
		}
		if (token.compare("Colors_Used") == 0)
		{
			colorsUsed = std::stoi(field);
		}
		if (token.compare("Color_Density") == 0)
		{
			colorDensity = std::stoi(field);
		}
		if (token.compare("Xcoord") == 0)
		{
			xcoord = std::stoi(field);
		}
		if (token.compare("Ycoord") == 0)
		{
			ycoord = std::stoi(field);
		}
		if (token.compare("Rzoom") == 0)
		{
			zoomFactor = std::stoi(field);
		}
		if (token.compare("InstructionSet") == 0)
		{
			instructionSet = std::stoi(field);
		}
		if (token.compare("Threads") == 0)
		{
			threadMax = std::stoi(field);
		}
		if (token.compare("ShadingMode") == 0)
		{
			shadingMode = std::stoi(field);
		}
		if (token.compare("ExecutionMode") == 0)
		{
			executionMode = std::stoi(field);
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

	xmin = (xcoord - ((double)w / h)*(zoomFactor));
	xmax = (xcoord + ((double)w / h)*(zoomFactor));
	ymin = (ycoord - (zoomFactor));
	ymax = (ycoord + (zoomFactor));

	//Initalize SDL
	if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
		std::cout << "SDL_Init Error: " << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_Window *window = SDL_CreateWindow("Mandelbrot Set", 100, 100, w, h,
		SDL_WINDOW_SHOWN);
	if (window == NULL) {
		std::cout << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
		return 2;
	}

	SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	SDL_Rect rect;
	rect.x = 0;
	rect.y = 0;
	rect.w = w;
	rect.h = h;

	//Generate from parameters
	fractalRender(renderer, rect, 1);

	bool exit = false;
	SDL_Event event;
	int mouseX;
	int mouseY;
	double centerX, centerY;
	double windowX, windowY;
	while (!exit)
	{
		while (SDL_WaitEvent(&event) != 0 && !exit) // SDL_WaitEvent can be used in place of polling, but will be stuck in this loop
		{
			switch (event.type)
			{
			case SDL_QUIT:
				exit = true;
				break;
			case SDL_KEYDOWN:
				switch (event.key.keysym.sym)
				{
				case SDLK_ESCAPE:
					std::cout << "Escape" << std::endl;
					exit = true;
					break;
				case SDLK_RETURN:
					writeBitmap(0, 0, w, h);
					std::cout << "Bitmap Created" << std::endl;
					break;

				case SDLK_UP:
					centerX = (xmax + xmin) / 2;
					centerY = (ymax + ymin) / 2;
					zoomFactor = zoomFactor * 1.05;
					xmin = (centerX - ((double)w / h)*(zoomFactor));
					xmax = (centerX + ((double)w / h)*(zoomFactor));
					ymin = (centerY - (zoomFactor));
					ymax = (centerY + (zoomFactor));
					fractalRender(renderer, rect, 0);
					std::cout << "Window is x:[" << xmin << ", " << xmax << "] y:[" << ymin << ", " << ymax << "]" << " r: " << zoomFactor << std::endl;
					break;

				case SDLK_DOWN:
					centerX = (xmax + xmin) / 2;
					centerY = (ymax + ymin) / 2;
					zoomFactor = zoomFactor / 1.05;
					xmin = (centerX - ((double)w / h)*(zoomFactor));
					xmax = (centerX + ((double)w / h)*(zoomFactor));
					ymin = (centerY - (zoomFactor));
					ymax = (centerY + (zoomFactor));
					fractalRender(renderer, rect, 0);
					std::cout << "Window is x:[" << xmin << ", " << xmax << "] y:[" << ymin << ", " << ymax << "]" << " r: " << zoomFactor << std::endl;
					break;

				default:
					break;
				}
			case SDL_MOUSEMOTION:
				SDL_GetMouseState(&mouseX, &mouseY);
				centerX = (xmax + xmin) / 2;
				centerY = (ymax + ymin) / 2;
				windowX = (xmax - xmin) / 2;
				windowY = (ymax - ymin) / 2;
				//std::cout << "MousePos at " << mouseX << " " << mouseY << std::endl;
				std::cout << "Point is " << centerX + windowX*((double)(mouseX - w / 2) * 2 / w) << " " << centerY + windowY*(-1)*((double)(mouseY - h / 2) * 2 / h);
				std::cout << " Escape Value is " << escapeMatrix[h*mouseY + mouseX];
				std::cout << " RGB is " << (int)screenBuffer[(3 * (h*mouseY + mouseX) + 2)] << " " << (int)screenBuffer[(3 * (h*mouseY + mouseX) + 1)] << " " << (int)screenBuffer[(3 * (h*mouseY + mouseX) + 0)] << std::endl;
				break;
			case SDL_MOUSEBUTTONUP:
				centerX = (xmax + xmin) / 2;
				centerY = (ymax + ymin) / 2;
				windowX = (xmax - xmin) / 2;
				windowY = (ymax - ymin) / 2;
				std::cout << "MouseUp on " << mouseX << " " << mouseY << std::endl;
				xmin = xmin + (windowX*((double)(mouseX - w / 2) * 2 / w));
				xmax = xmax + (windowX*((double)(mouseX - w / 2) * 2 / w));
				ymin = ymin + (windowY*(-1)*((double)(mouseY - h / 2) * 2 / h));
				ymax = ymax + (windowY*(-1)*((double)(mouseY - h / 2) * 2 / h));
				fractalRender(renderer, rect, 1);
				std::cout << "Window is x:[" << xmin << ", " << xmax << "] y:[" << ymin << ", " << ymax << "] " << "r: " << zoomFactor << std::endl;
				break;
			case SDL_MOUSEWHEEL:
				std::cout << "MouseWheel Dir " << event.wheel.y << std::endl;
				centerX = (xmax + xmin) / 2;
				centerY = (ymax + ymin) / 2;
				if (event.wheel.y == 1)
				{
					zoomFactor = zoomFactor / 2;
				}
				else if (event.wheel.y == -1)
				{
					zoomFactor = zoomFactor * 2;
				}
				xmin = (centerX - ((double)w / h)*(zoomFactor));
				xmax = (centerX + ((double)w / h)*(zoomFactor));
				ymin = (centerY - (zoomFactor));
				ymax = (centerY + (zoomFactor));
				fractalRender(renderer, rect, 1);
				std::cout << "Window is x:[" << xmin << ", " << xmax << "] y:[" << ymin << ", " << ymax << "]" << " r: " << zoomFactor << std::endl;
				break;
			}
			//SDL_Delay(10); // Use for PollEvent
		}
	}

	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}

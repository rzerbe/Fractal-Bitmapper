#include <cstdio>
#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>
#include <stdint.h>
#include <string>

/*
ctrl-F "objective" to find potential things to work on, in no particular order.

objective1:
Redesign our fractal() function to be called each time wedo the computation for a new image.
Take some of what we have in the main() function and design a
render() function that converts the escapeMatrix to a colored pixel array.
Easily generate multiple images with a single run of this program (useful for getting various magnifications).

objective2:
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
const int w = 2000;
const int h = 2000;

struct color
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

/* The state must be seeded so that it is not everywhere zero. */
uint64_t s[2];

uint64_t xorshift128plus(void) {
	uint64_t x = s[0];
	uint64_t const y = s[1];
	s[0] = y;
	x ^= x << 23; // a
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
	return s[1] + y;
}

const int upperBound = 10000;
const int escapeRadiusSquared = (1 << 16);		//2^16
int maxIter = 255;
const int numColors = 9;
int colorDensity;
double zr[upperBound];
double zi[upperBound];
int colorsUsed;
std::string multiple;
double zoomFactor;
int numImages;

std::vector<double> escapeMatrix;	//escape matrix now holds doubles to accomodate smooth coloring


void fractal()
{
	double xmax, xmin, ymax, ymin, xcoord, ycoord, windowWidth, windowHeight;
	std::string choice;
	escapeMatrix.reserve(w*h);
	std::cout << "Enter maximum number of iterations\n";
	std::cin >> maxIter;
	std::cout << "Enter a point to be centered on, or a window? point/window\n";
	std::cin >> choice;

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
		*/
		std::cout << "Do you wish to generate multiple images? y/n\n";
		std::cin >> multiple;
		std::cout << "Enter zoom factor:\n";
		std::cin >> zoomFactor;
		std::cout << "Enter number of images to generate:\n";
		std::cin >> numImages;
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

	std::cout << "Enter number of colors to use:\n";
	std::cin >> colorsUsed;
	std::cout << "Enter color density:\n";
	std::cin >> colorDensity;

	for (int i = 0; i < w; i++)
	{
		//give percentage progress
		int m = i % (w / 100);
		if (m == 0)
			std::cout << "Computation " << (int)(((double)i / w * 100)) << "% complete.\n";

		double cr = xmin + ((double)i / w) * (xmax - xmin);
		for (int j = 0; j < h; j++)
		{
			bool shortcut = false;
			double ci = ymax - ((double)j / h) * (ymax - ymin);

			//check whether point lies in main cardioid
			double q;
			q = (cr - 0.25)*(cr - 0.25) + ci*ci;
			if (q*(q + cr - 0.25) - 0.25*ci*ci < 0)
			{
				escapeMatrix.emplace_back(maxIter);
				shortcut = true;
			}

			//check whether point lies in period-2 bulb
			if ((cr + 1)*(cr + 1) + ci*ci - 0.0625 < 0)
			{
				escapeMatrix.emplace_back(maxIter);
				shortcut = true;
			}

			//if point lies within main cardioid or bulb, skip computation
			if (!shortcut)
			{
				int k;
				double logzMag, zMagSqr, potential;
				zr[0] = 0;
				zi[0] = 0;

				for (k = 1; k <= maxIter; k++)
				{

					zr[k] = pow(zr[k - 1], 2) - pow(zi[k - 1], 2) + cr;
					zi[k] = 2 * zr[k - 1] * zi[k - 1] + ci;
					zMagSqr = zr[k] * zr[k] + zi[k] * zi[k];
					if (zMagSqr > escapeRadiusSquared)		//did some optimizations in this area
					{
						break;
					}
				}
				//Non-convergence (k was stored as -2^31 - 1 if not handled)
				if (k == maxIter + 1)
				{
					escapeMatrix.emplace_back(maxIter);
				}
				else // Converges before maxIter
				{
					double dub;
					logzMag = abs(log(zMagSqr) / 2);
					//potential = log(logzMag / log(2)) / log(2);
					potential = log(logzMag) / log(2);
					dub = k + 1 - potential;			//this result is a double
					if (dub > maxIter || dub < -1000)
						dub = maxIter;
					escapeMatrix.emplace_back(dub);
				}
			}
		}
	}
}


int main()
{
	s[0] = std::time(NULL);
	s[1] = 25;

	FILE *f;
	unsigned char *img = NULL;
	int filesize = 54 + 3 * w*h;  //w is your image width, h is image height, both int
								  /*if (img)
								  free(img);
								  img = (unsigned char *)malloc(3 * w*h);
								  memset(img, 0, sizeof(img));*/

	unsigned char bmpfileheader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
	unsigned char bmpinfoheader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0 };
	unsigned char bmppad[3] = { 0,0,0 };

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

	f = fopen("img1.bmp", "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);


	fractal(); // Generate mandelbutt set

	int x = 0;
	int y = 0;

	//colorDraw is the color we will draw with
	//color1 and color2 are the colors that need to be linearly interpolated to get colorDraw
	//palette stores the rgb of all of our hardcoded colors
	unsigned char colorDraw[3];
	unsigned char color1[3];
	unsigned char color2[3];
	unsigned char palette[numColors][3];


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


	for (int i = 0; i<w; i++)
	{
		//give percentage progress
		int m = i % (w / 100);
		if (m == 0)
			std::cout << "Coloring " << (int)(((double)i / w * 100)) << "% complete.\n";
			
		for (int j = 0; j<h; j++)
		{

			if (escapeMatrix.at(j*h + i) == maxIter)
			{
				colorDraw[0] = 0;
				colorDraw[1] = 0;
				colorDraw[2] = 0;
			}
			else
			{
				double tempIter = maxIter;
				double k = escapeMatrix.at(j*h + i);


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
				colorDraw[0] = (color2[0] - color1[0])*(kscaled - floor(kscaled)) + color1[0];
				colorDraw[1] = (color2[1] - color1[1])*(kscaled - floor(kscaled)) + color1[1];
				colorDraw[2] = (color2[2] - color1[2])*(kscaled - floor(kscaled)) + color1[2];


				//Monochromatic
				/*r[0] = 255 - k;
				g[0] = 255 - k;
				b[0] = 255 - k;*/
			}

			fwrite(&colorDraw[2], 1, 1, f);
			fwrite(&colorDraw[1], 1, 1, f);
			fwrite(&colorDraw[0], 1, 1, f);

		}

		fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, f);

		/*
		fwrite(b, 1, 1, f);
		fwrite(g, 1, 1, f);
		fwrite(r, 1, 1, f);
		*/
		//I replaced the above code with this, but there is some sort of error
		//Fix it senpai >.<

		/*img[(x + y*w) * 3 + 2] = (unsigned char)(r[0]);
		img[(x + y*w) * 3 + 1] = (unsigned char)(g[0]);
		img[(x + y*w) * 3 + 0] = (unsigned char)(b[0]);*/
	}
	/*for (int i = 0; i<h; i++)
	{
	fwrite(img + (w*(h - i - 1) * 3), 3, w, f);
	fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, f);
	}*/
	fclose(f);

	system("pause");
	return 0;
}

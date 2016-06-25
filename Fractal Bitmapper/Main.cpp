#include <cstdio>
#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>
#include <stdint.h>
#include <string>

//apparently 30000 is too much
const int w = 12000;
const int h = 12000;

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
double zr[upperBound];
double zi[upperBound];

std::vector<double> escapeMatrix;	//escape matrix now holds doubles to accomodate smooth coloring


double colorsUsed;

void fractal()
{
	double xmax, xmin, ymax, ymin, xcoord, ycoord, width, height;
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
		std::cout << "Enter width:\n";
		std::cin >> width;
		std::cout << "Enter height:\n";
		std::cin >> height;

		xmin = xcoord - width / 2;
		xmax = xcoord + width / 2;
		ymin = ycoord - height / 2;
		ymax = ycoord + height / 2;
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

	for (int i = 0; i < w; i++)
	{
		//give percentage progress
		int m = i % (w / 100);
		if (m == 0)
			std::cout << (int)(((double)i / w * 100)) << "% complete.\n";

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

	//black
	palette[8][0] = 0;
	palette[8][1] = 0;
	palette[8][2] = 0;


	for (int i = 0; i<w; i++)
	{
		for (int j = 0; j<h; j++)
		{
			//x = i;
			//y = (h - 1) - j;

			//r = (unsigned char)(xorshift128plus() % 255);
			//g = (unsigned char)(xorshift128plus() % 255);
			//b = (unsigned char)(xorshift128plus() % 255);

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

				//this code is obsolete
				/*
				color[0] = (((int)k % (int)(tempIter/(numColors-2))) / (tempIter/(numColors-2))) *
				(r[(int)ceil(k / tempIter * (numColors - 2)) + 1] - r[(int)ceil(k / tempIter * (numColors-2))]) +
				r[(int)ceil(k / tempIter * (numColors - 2))];
				color[1] = (((int)k % (int)(tempIter / (numColors - 2))) / (tempIter / (numColors - 2))) *
				(g[(int)ceil(k / tempIter * (numColors - 2)) + 1] - g[(int)ceil(k / tempIter * (numColors - 2))]) +
				g[(int)ceil(k / tempIter * (numColors - 2))];
				color[2] = (((int)k % (int)(tempIter / (numColors - 2))) / (tempIter / (numColors - 2))) *
				(b[(int)ceil(k / tempIter * (numColors - 2)) + 1] - b[(int)ceil(k / tempIter * (numColors - 2))]) +
				b[(int)ceil(k / tempIter * (numColors - 2))];
				*/


				/*
				I changed a lot.  There are probably bugs.  What we need to figure out is
				what the ballpark values of k (escapeMatrix) come out to be.  This code tries to generate
				a color based on the escapeMatrix value, so the escapeMatrix values should be from 0 to
				the number of colors in the palette minus 1.  So, if there are 10 colors in the palette,
				and we have a pixel in the main part of the mandelbrot set (the huge black bulb in the middle),
				then the k value should be 9, since palette[9] gives the rgb of the color black.  As we get
				values lower than 9, we will linearly interpolate to the other colors.
				*/


				//colorDraw is the color we will draw with
				//color1 and color2 are the colors that need to be linearly interpolated to get colorDraw

				//figure out which two colors our point should be interpolated between, based on k
				double kscaled = k*(colorsUsed - 1) / maxIter;
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

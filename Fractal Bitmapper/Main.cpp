#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>
#include <stdint.h>
#include <string>

#include <thread>

//#include <GL/glew.h>
//#include <GLFW/glfw3.h>
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
const int w = 2560;
const int h = 1440;

const int upperBound = 10000;
const int escapeRadiusSquared = (1 << 16);		//2^16
int maxIter = 256;
const int numColors = 8;
int colorDensity;
int colorsUsed;
std::string multiple;
double zoomFactor;
int numImages;

double xmin, xmax, ymin, ymax;

//std::vector<double> escapeMatrix;	//escape matrix now holds doubles to accomodate smooth coloring
double escapeMatrix[w*h];
void adjustParameters()
{
	std::string choice;
	//escapeMatrix.reserve(w*h + h);
	//escapeMatrix.resize(w*h + h);
	std::cout << "Enter maximum number of iterations\n";
	std::cin >> maxIter;
	std::cout << "Enter vertical resolution\n";
	//std::cin >> h;
	std::cout << "Enter horizontal resolution\n";
	//std::cin >> w;
	std::cout << "Enter a point to be centered on, or a window? point/standardpt/window\n";
	std::cin >> choice;

	if (choice == "standardpt")
	{
		double xcoord, ycoord, windowWidth, windowHeight;
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
		double xcoord, ycoord, windowWidth, windowHeight;
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

	std::cout << "Enter number of colors to use " << "(between 1 and " << numColors << "):\n";
	std::cin >> colorsUsed;
	std::cout << "Enter color density (how many iterations it takes to cycle through the whole palette):\n";
	std::cin >> colorDensity;
}

void fractal(int i_init, int j_init, int i_fin, int j_fin, int threadID, int maxThreads)
{
	double zr[upperBound];
	double zi[upperBound];
	time_t start, end;
	time(&start);
	int m;
	double ci, cr, q, logzMag, zMagSqr, potential;
	bool shortcut;

	for (int j = j_init; j < j_fin; j++) // y axis
	{
		//give percentage progress
		m = j % (h / 100);
		if (m == 0)
			std::cout << "Thread: " << threadID << " Computation " << (int)(((double)(j-j_init) / (j_fin-j_init) * 100)) << "% complete.\n";
		ci = ymin + ((double)j / h) * (ymax - ymin);
		
		for (int i = i_init; i < i_fin; i++) // x axis
		{
			shortcut = false;
			cr = xmin + ((double)i / w) * (xmax - xmin);

			//check whether point lies in main cardioid
			q = (cr - 0.25)*(cr - 0.25) + ci*ci;
			if (q*(q + cr - 0.25) - 0.25*ci*ci < 0)
			{
				escapeMatrix[j*w + i] = maxIter;
				shortcut = true;
			}
			//check whether point lies in period-2 bulb
			else if ((cr + 1)*(cr + 1) + ci*ci - 0.0625 < 0)
			{
				escapeMatrix[j*w + i] = maxIter;
				shortcut = true;
			}

			//if point lies within main cardioid or bulb, skip computation
			if (!shortcut)
			{
				int k;
				zr[0] = 0;
				zi[0] = 0;

				for (k = 1; k < maxIter; k++)
				{

					zr[k] = (zr[k - 1]*zr[k - 1] - zi[k - 1]*zi[k - 1]) + cr; // heavily optimized by removing pow
					zi[k] = 2 * zr[k - 1] * zi[k - 1] + ci;
					zMagSqr = zr[k] * zr[k] + zi[k] * zi[k];
					if (zMagSqr > escapeRadiusSquared)		//did some optimizations in this area
					{
						break;
					}
				}
				//Non-convergence (k was stored as -2^31 - 1 if not handled)
				if (k == maxIter)
				{
					escapeMatrix[j*w + i] = maxIter;
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
					escapeMatrix[j*w + i] = dub;
				}
			}
		}
	}
	time(&end);
	std::cout << std::endl << std::endl << "Thread " << threadID << " finished in " << difftime(end, start) << " seconds" << std::endl << std::endl;
}

void writeBitmap(FILE* bitmap, int i_init, int j_init, int i_fin, int j_fin)
{
	//colorDraw is the color we will draw with
	//color1 and color2 are the colors that need to be linearly interpolated to get colorDraw
	//palette stores the rgb of all of our hardcoded colors
	unsigned char colorDraw[3*w];
	unsigned char color1[3];
	unsigned char color2[3];
	unsigned char palette[numColors][3];
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
		int m = j % (h / 100);
		if (m == 0)
			std::cout << "Coloring " << (int)(((double)(j) / h * 100)) << "% complete.\n";

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
				double tempIter = maxIter;
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

			//fwrite(&colorDraw[0], 1, 1, bitmap);
			//fwrite(&colorDraw[1], 1, 1, bitmap);
			//fwrite(&colorDraw[2], 1, 1, bitmap);
			//fwrite(colorDraw, 1, 3 * w, bitmap);
		}
		fwrite(&colorDraw, 1, 3 * w, bitmap);
		fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, bitmap);
	}
}
const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 480;
int main(int argc, char** argv)
{
	// Initialise GLFW
	/*if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); //We don't want the old OpenGL 

																   // Open a window and create its OpenGL context
	GLFWwindow* window; // (In the accompanying source code, this variable is global)
	window = glfwCreateWindow(1024, 768, "Tutorial 01", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window); // Initialize GLEW
	glewExperimental = true; // Needed in core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}


	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	glEnable(GL_PROGRAM_POINT_SIZE);
	glPointSize(1.0);

	do {
		// Draw nothing, see you in tutorial 2 !

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

	} // Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0);
		*/


	//Bitmap Header
	FILE *bitmap;
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

	bitmap = fopen("img1.bmp", "wb");
	fwrite(bmpfileheader, 1, 14, bitmap);
	fwrite(bmpinfoheader, 1, 40, bitmap);

	//User Input
	adjustParameters();

	//CPU Multithreading
	time_t start, end;
	const int threadMax = 4;

	std::thread fractalT[threadMax];
	int thread = 0;
	time(&start);

	for (int y = 0; y < threadMax / 2; ++y) // Starting y subdivison
	{
		for (int x = 0; x < (threadMax / 2); ++x) // Starting x subdivison
		{
			if (thread == threadMax - 1) // Allow main thread to perform computation
			{
				std::cout << "Thread Main " << thread << " initialized with parameters"
					<< " xinit " << (w*x) / (threadMax / 2)
					<< " yinit " << (h*y) / (threadMax / 2)
					<< " xfin " << (w*(x + 1)) / (threadMax / 2)
					<< " yfin " << (h*(y + 1)) / (threadMax / 2) << std::endl;
				fractal(w / (threadMax / 2), h / (threadMax / 2), w, h, threadMax - 1, threadMax);
			}
			else // Create new thread
			{
				if (x == threadMax / 2)
				{
					fractalT[thread] = std::thread(fractal, (w*x) / (threadMax / 2), (h*y) / (threadMax / 2), (w*(x + 1)) / (threadMax / 2) + (w - 2 * (w / (threadMax / 2))), (h*(y + 1)) / (threadMax / 2), thread, threadMax);
					std::cout << "Thread " << thread << " initialized with parameters"
						<< " xinit " << (w*x) / (threadMax / 2)
						<< " yinit " << (h*y) / (threadMax / 2)
						<< " xfin " << (w*(x + 1)) / (threadMax / 2) + (w - 2 * (w / (threadMax / 2)))
						<< " yfin " << (h*(y + 1)) / (threadMax / 2) << std::endl;
					thread++;
				}
				else if (y == threadMax / 2)
				{
					fractalT[thread] = std::thread(fractal, (w*x) / (threadMax / 2), (h*y) / (threadMax / 2), (w*(x + 1)) / (threadMax / 2), (h*(y + 1)) / (threadMax / 2) + (h - 2 * (h / (threadMax / 2))), thread, threadMax);
					std::cout << "Thread " << thread << " initialized with parameters"
						<< " xinit " << (w*x) / (threadMax / 2)
						<< " yinit " << (h*y) / (threadMax / 2)
						<< " xfin " << (w*(x + 1)) / (threadMax / 2)
						<< " yfin " << (h*(y + 1)) / (threadMax / 2) + (h - 2 * (h / (threadMax / 2))) << std::endl;
					thread++;
				}
				else
				{
					fractalT[thread] = std::thread(fractal, (w*x) / (threadMax / 2), (h*y) / (threadMax / 2), (w*(x + 1)) / (threadMax / 2), (h*(y + 1)) / (threadMax / 2), thread, threadMax);
					std::cout << "Thread " << thread << " initialized with parameters"
						<< " xinit " << (w*x) / (threadMax / 2)
						<< " yinit " << (h*y) / (threadMax / 2)
						<< " xfin " << (w*(x + 1)) / (threadMax / 2)
						<< " yfin " << (h*(y + 1)) / (threadMax / 2) << std::endl;
					thread++;
				}
			}
		}
	}

	for (int i = 0; i < threadMax - 1; ++i)
	{
		fractalT[i].join();
	}
	time(&end);
	std::cout << "Fractal Generation finished in " << difftime(end, start) << " seconds" << std::endl;

	writeBitmap(bitmap, 0, 0, w, h);
	fclose(bitmap);




	if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
		std::cout << "SDL_Init Error: " << SDL_GetError() << std::endl;
		return 1;
	}

	SDL_Window *window = SDL_CreateWindow("Mandelbrot Set", 0, 0, w, h,
		SDL_WINDOW_SHOWN);
	if (window == NULL) {
		std::cout << "SDL_CreateWindow Error: " << SDL_GetError() << std::endl;
		return 2;
	}
	/*
	SDL_Renderer *ren = SDL_CreateRenderer(window, -1,
	SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (ren == NULL){
	std::cout << "SDL_CreateRenderer Error: " << SDL_GetError() << std::endl;
	return 1;
	}
	*/
	SDL_Surface *surface_bmp = SDL_LoadBMP("img1.bmp");
	if (surface_bmp == NULL) {
		std::cout << "SDL_LoadBMP Error: " << SDL_GetError() << std::endl;
		return 3;
	}
	/*
	SDL_Texture *tex = SDL_CreateTextureFromSurface(ren, surface_bmp);
	SDL_FreeSurface(surface_bmp);
	if (tex == NULL){
	std::cout << "SDL_CreateTextureFromSurface Error: "
	<< SDL_GetError() << std::endl;
	return 1;
	}
	*/

	SDL_Surface *surface_window = SDL_GetWindowSurface(window);

	/*
	* Copies the bmp surface to the window surface
	*/
	SDL_BlitSurface(surface_bmp,
		NULL,
		surface_window,
		NULL);

	/*
	* Now updating the window
	*/
	SDL_UpdateWindowSurface(window);

	/*
	* This function used to update a window with OpenGL rendering.
	* This is used with double-buffered OpenGL contexts, which are the default.
	*/
	/*    SDL_GL_SwapWindow(window); */


	/*    SDL_DestroyTexture(tex);*/
	/*    SDL_DestroyRenderer(ren);*/

	system("pause");
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}

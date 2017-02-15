#pragma once
struct Configuration
{
	bool renderCPU = 1;
	bool renderGPU = 1;

	int maxIter = 256;
	int paletteNumber = 8;
	int colorDensity = 256;
	int colorsUsed = 8;
	double zoomFactor = 1;
	int shadingMode = 0;
	bool smoothShading = 0;
	int threadMax = 1;
	int instructionSet = 0;
	int executionMode = 0;
	int SSAA = 1;

	double zoom_coarse = 2;
	double zoom_fine = 1.1;
	double zoom_wheel = 1.5;

	std::string paletteFile = "";

	int resolutionX = 512;
	int resolutionY = 512;

	int bufferX = 512;
	int bufferY = 512;

	unsigned char *palette;

	double xmin, ymin = -1;
	double xmax, ymax = 1;
	double xcoord, ycoord, windowWidth, windowHeight;
	double centerX, centerY, windowX, windowY;
};

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
	int threadMax = 4;
	int instructionSet = 0;
	int executionMode = 0;
	std::string paletteFile = "";

	int resolutionX = 512;
	int resolutionY = 512;

	unsigned char *palette;

	double xmin, ymin = -1;
	double xmax, ymax = 1;
	double xcoord, ycoord, windowWidth, windowHeight;
	double centerX, centerY, windowX, windowY;
};

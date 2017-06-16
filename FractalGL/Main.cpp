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

// Renderer
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>

// Program Structure
#include "Configuration.h"
#include "Calculation.h"
#include "Command.h"
#include "F_Mandelbrot.h"
#include "F_Buddhabrot.h"
#include "Input.h"

////////////////////////////////////////
// Program Data
Configuration config;
Calculation_Data calculation;
F_Mandelbrot *f_mandelbrot;
F_Buddhabrot *f_buddhabrot;
std::vector<Command*> commandQueue;

// Graphics
GLuint gl_PBO, gl_Tex, gl_Shader;
GLFWwindow* window;
////////////////////////////////////////

void display_func();
void command_prompt_poll();
void command_prompt_enqueue(int type);
void resize(GLFWwindow* window, int width, int height);
bool initialize_buffers(int resolutionX, int resolutionY, bool display);
void delete_buffers();

// Mandelbrot
void render_optimizer(int iterations, int delta)
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

void action_zoom(double factor)
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
	f_mandelbrot->fractal_render(0);
	display_func();
	printf("Window is x:[%9e, %9e] y:[%9e, %9e] r:[%9e] \n",
		config.xmin, config.xmax,
		config.ymin, config.ymax,
		config.zoomFactor
		);
}

void action_point(double centerX, double centerY, double factor)
{
	config.zoomFactor = factor;
	config.xmin = (centerX - ((double)config.resolutionX / config.resolutionY)*(config.zoomFactor));
	config.xmax = (centerX + ((double)config.resolutionX / config.resolutionY)*(config.zoomFactor));
	config.ymin = (centerY - (config.zoomFactor));
	config.ymax = (centerY + (config.zoomFactor));
	memset(calculation.updatePixel, true, config.resolutionX * config.resolutionY);
	calculation.screenStill = 0;
	command_prompt_enqueue(0);
	printf("Window is x:[%9e, %9e] y:[%9e, %9e] r:[%9e] \n",
		config.xmin, config.xmax,
		config.ymin, config.ymax,
		config.zoomFactor
		);
}

void action_ssaa(int n)
{
	memset(calculation.updatePixel, true, config.resolutionX * config.resolutionY);

	config.SSAA = n;

	calculation.screenStill = 0;
	command_prompt_enqueue(2);
	printf("SSAA is %d\n",
		config.SSAA
	);
}

void action_iterations(bool addOrMult, double factor)
{
	if (addOrMult == 0)
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
	render_optimizer(config.maxIter, config.maxIter - calculation.prevMaxIter);
	calculation.screenStill = 1;
	command_prompt_enqueue(1);
	printf("Maximum iterations %d \n", config.maxIter);
}

void action_buddha()
{
	f_buddhabrot = new F_Buddhabrot(calculation, config);
	calculation.screenStill = 0;
	command_prompt_enqueue(3);
}

void print_error(int err, const char* msg)
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
				action_zoom(1.0/config.zoom_coarse);
				break;
			case 'A': /// Decrease iterations
				action_iterations(0, 0.5);
				break;
			case 'S': /// Zoom out
				action_zoom(config.zoom_coarse);
				break;
			case 'D': /// Increase iterations
				action_iterations(0, 2);
				break;
			case GLFW_KEY_LEFT:
			{
				double delta = (config.xmax - config.xmin);
				config.xmax = config.xmax - (delta) * (0.1);
				config.xmin = config.xmin - (delta) * (0.1);
				calculation.screenStill = 0;
				memset(calculation.updatePixel, true, config.resolutionX * config.resolutionY);
				f_mandelbrot->fractal_render(0);
				display_func();
				break;
			}
			case GLFW_KEY_RIGHT:
			{
				double delta = (config.xmax - config.xmin);
				config.xmax = config.xmax + (delta) * (0.1);
				config.xmin = config.xmin + (delta) * (0.1);
				calculation.screenStill = 0;
				memset(calculation.updatePixel, true, config.resolutionX * config.resolutionY);
				f_mandelbrot->fractal_render(0);
				display_func();
				break;
			}
			case GLFW_KEY_UP:
			{
				double delta = (config.ymax - config.ymin);
				config.ymax = config.ymax - (delta) * (0.1);
				config.ymin = config.ymin - (delta) * (0.1);
				calculation.screenStill = 0;
				memset(calculation.updatePixel, true, config.resolutionX * config.resolutionY);
				f_mandelbrot->fractal_render(0);
				display_func();
				break;
			}
			case GLFW_KEY_DOWN:
			{
				double delta = (config.ymax - config.ymin);
				config.ymax = config.ymax + (delta) * (0.1);
				config.ymin = config.ymin + (delta) * (0.1);
				calculation.screenStill = 0;
				memset(calculation.updatePixel, true, config.resolutionX * config.resolutionY);
				f_mandelbrot->fractal_render(0);
				display_func();
				break;
			}

			case 'Q': /// Increase iterations auto
				for (int renderNumber = 0; renderNumber < 256; ++renderNumber)
				{
					calculation.prevMaxIter = config.maxIter;
					config.maxIter = config.maxIter + 1;
					render_optimizer(config.maxIter, config.maxIter - calculation.prevMaxIter);
					calculation.screenStill = 1;
					f_mandelbrot->fractal_render(0);
					display_func();
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
					printf("Render Optimization: On\n");
				}
				else
				{
					calculation.screenOptimization = 0;
					printf("Render Optimization: Off\n");
				}
				break;
			case GLFW_KEY_ESCAPE: /// Quit program
				glfwSetWindowShouldClose(window, GLFW_TRUE);
				delete_buffers();
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
				action_zoom(1.0/config.zoom_fine);
				break;
			case 'A': /// Decrease iterations
				action_iterations(1, -1);
				break;
			case 'S': /// Zoom out
				action_zoom(config.zoom_fine);
				break;
			case 'D': /// Increase iterations
				action_iterations(1, 1);
				break;
			default:
				break;
			}

		}
	}
}

void mouse_click(GLFWwindow* window, int button, int action, int mods)
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
			f_mandelbrot->fractal_render(1);
			display_func();
			printf("Window is x:[%9e, %9e] y:[%9e, %9e] r:[%9e] \n",
				config.xmin, config.xmax,
				config.ymin, config.ymax,
				config.zoomFactor
				);
	}
}

void mouse_scroll(GLFWwindow* window, double scrollX, double scrollY)
{
	if (scrollY > 0) // Scroll forward
	{
		action_zoom(1.0/config.zoom_wheel);
	}
	else if (scrollY < 0) // Scroll back
	{
		action_zoom(config.zoom_wheel);
	}
}

void mouse_motion(GLFWwindow* window, double x, double y)
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
		(int)calculation.screenBufferCPU[(4 * (config.resolutionX*mouseY + mouseX) + 2)], 
		(int)calculation.screenBufferCPU[(4 * (config.resolutionX*mouseY + mouseX) + 1)], 
		(int)calculation.screenBufferCPU[(4 * (config.resolutionX*mouseY + mouseX) + 0)],
		(int)calculation.updatePixel[config.resolutionX*mouseY + mouseX]
		);

	//double fx = (double)(x - lastx) / 50.0 / (double)(imageW);
	//double fy = (double)(lasty - y) / 50.0 / (double)(imageH);
}

// Rendering
void display_func()
{
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, config.resolutionX, config.resolutionY, GL_BGRA, GL_UNSIGNED_BYTE, (GLvoid*)calculation.screenBufferCPU);

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
	//glDisable(GL_FRAGMENT_PROGRAM_ARB);

	glfwSwapBuffers(window);
}

void output_bitmap(std::string filename)
{
	FILE *f = NULL;
	uint32_t filesize = 54 + 3 * config.resolutionX*config.resolutionY;

	unsigned char bmpfileheader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
	unsigned char bmpinfoheader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 32,0 };
	unsigned char bmppad[4] = { 0,0,0,0 };

	bmpfileheader[2] = (unsigned char)(filesize);
	bmpfileheader[3] = (unsigned char)(filesize >> 8);
	bmpfileheader[4] = (unsigned char)(filesize >> 16);
	bmpfileheader[5] = (unsigned char)(filesize >> 24);

	bmpinfoheader[4] = (unsigned char)(config.resolutionX);
	bmpinfoheader[5] = (unsigned char)(config.resolutionX >> 8);
	bmpinfoheader[6] = (unsigned char)(config.resolutionX >> 16);
	bmpinfoheader[7] = (unsigned char)(config.resolutionX >> 24);
	bmpinfoheader[8] = (unsigned char)(config.resolutionY);
	bmpinfoheader[9] = (unsigned char)(config.resolutionY >> 8);
	bmpinfoheader[10] = (unsigned char)(config.resolutionY >> 16);
	bmpinfoheader[11] = (unsigned char)(config.resolutionY >> 24);

	fopen_s(&f, filename.c_str(), "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);

	for (int j = 0; j < config.resolutionY; j++)
	{
		fwrite((calculation.screenBufferCPU+4*j*config.resolutionX), sizeof(char), 4 * config.resolutionX, f);
		//fwrite(bmppad, 1, (4 - (j * 3) % 4) % 4, f);
	}
	fclose(f);
}

void output_render(std::string filename, std::string fractal, int resolutionX, int resolutionY)
{
	auto begin = std::chrono::high_resolution_clock::now();

	Configuration tempConfig;
	Calculation_Data tempCalculation;

	tempConfig = config;
	tempCalculation = calculation;

	tempConfig.resolutionX = resolutionX;
	tempConfig.resolutionY = resolutionY;
	tempConfig.bufferX = resolutionX * tempConfig.SSAA;
	tempConfig.bufferY = resolutionY * tempConfig.SSAA;
	
	tempCalculation.escapeBufferCPU = NULL;
	tempCalculation.escapeBufferSuperSampling = NULL;
	tempCalculation.magnitude = NULL;
	tempCalculation.updatePixel = NULL;
	tempCalculation.screenBufferCPU = new unsigned char[resolutionX * resolutionY * 4];

	if (fractal.compare("mandelbrot") == 0)
	{
		F_Mandelbrot *temp_f_mandelbrot = new F_Mandelbrot(tempCalculation, tempConfig);
		tempCalculation.escapeBufferCPU = new double[resolutionX * resolutionY];
		tempCalculation.escapeBufferSuperSampling = new double[resolutionX * resolutionY * tempConfig.SSAA * tempConfig.SSAA];
		tempCalculation.magnitude = new double[resolutionX * resolutionY];
		tempCalculation.updatePixel = new bool[resolutionX * resolutionY];
		temp_f_mandelbrot->fractal_render(1);
		delete temp_f_mandelbrot;
	}
	else if (fractal.compare("buddhabrot") == 0)
	{
		F_Buddhabrot *temp_f_buddhabrot = new F_Buddhabrot(tempCalculation, tempConfig);
		temp_f_buddhabrot->fractal_render(1);
		delete temp_f_buddhabrot;
	}

	FILE *f = NULL;
	uint32_t filesize = 54 + 3 * resolutionX * resolutionY;

	uint8_t bmpfileheader[14] = { 'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0 };
	uint8_t bmpinfoheader[40] = { 40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 32,0 };
	uint8_t bmppad[4] = { 0,0,0,0 };

	bmpfileheader[2] = (uint8_t)(filesize);
	bmpfileheader[3] = (uint8_t)(filesize >> 8);
	bmpfileheader[4] = (uint8_t)(filesize >> 16);
	bmpfileheader[5] = (uint8_t)(filesize >> 24);

	bmpinfoheader[4] = (uint8_t)(tempConfig.resolutionX);
	bmpinfoheader[5] = (uint8_t)(tempConfig.resolutionX >> 8);
	bmpinfoheader[6] = (uint8_t)(tempConfig.resolutionX >> 16);
	bmpinfoheader[7] = (uint8_t)(tempConfig.resolutionX >> 24);
	bmpinfoheader[8] = (uint8_t)(tempConfig.resolutionY);
	bmpinfoheader[9] = (uint8_t)(tempConfig.resolutionY >> 8);
	bmpinfoheader[10] = (uint8_t)(tempConfig.resolutionY >> 16);
	bmpinfoheader[11] = (uint8_t)(tempConfig.resolutionY >> 24);

	fopen_s(&f, filename.c_str(), "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);

	for (int j = 0; j < resolutionY; j++)
	{
		fwrite((tempCalculation.screenBufferCPU + 4 * j * resolutionX), sizeof(char), 4 * resolutionX, f);
		//fwrite(bmppad, 1, (4 - (j * 3) % 4) % 4, f);
	}
	fclose(f);

	delete tempCalculation.screenBufferCPU;

	delete tempCalculation.escapeBufferCPU;
	delete tempCalculation.escapeBufferSuperSampling;
	delete tempCalculation.magnitude;
	delete tempCalculation.updatePixel;

	auto end = std::chrono::high_resolution_clock::now();
	auto elasped = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	printf("Write Render Time %d milliseconds\n",
		elasped
	);
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

void delete_buffers()
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

	if (calculation.escapeBufferSuperSampling)
	{
		delete[] calculation.escapeBufferSuperSampling;
		calculation.escapeBufferSuperSampling = NULL;
	}

	if (calculation.updatePixel)
	{
		delete[] calculation.updatePixel;
		calculation.updatePixel = NULL;
	}

	if (calculation.magnitude)
	{
		delete[] calculation.magnitude;
		calculation.magnitude = NULL;
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
	printf("Deleted\n");
}

bool initialize_buffers(int width, int height, bool display)
{
	// Flush buffers
	if (display == true)
	{
		delete_buffers();
	}

	// Check for minimized window
	if ((width == 0) && (height == 0))
	{
		return false;
	}

	// Allocate Buffers
	calculation.escapeBufferCPU = new double[width * height];
	calculation.escapeBufferSuperSampling = new double[width * height * config.SSAA * config.SSAA];
	calculation.magnitude = new double[width * height * config.SSAA * config.SSAA];
	calculation.screenBufferCPU = new unsigned char[width * height * 4];
	calculation.updatePixel = new bool[width * height];
	memset(calculation.updatePixel, true, width * height);
	std::cout << "Resolution Set to " << width << " " << height << std::endl;
	std::cout << "Buffer Set to " << width * config.SSAA << " " << height * config.SSAA << std::endl;

	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, (GLvoid*)calculation.screenBufferCPU);
	glEnable(GL_TEXTURE_2D);

	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, (GLvoid*)calculation.screenBufferCPU, GL_STREAM_COPY);

	gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, "!!ARBfp1.0\n"
		"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
		"END");

	if (display == true) // Display
	{
		display_func();
		display_func();
	}
	return true;
}

void resize(GLFWwindow* window, int width, int height)
{
	if (width != config.resolutionX && height != config.resolutionY)
	{
		initialize_buffers(width, height, true);
		config.resolutionX = width;
		config.resolutionY = height;
		config.bufferX = width * config.SSAA;
		config.bufferY = height * config.SSAA;

		glViewport(0, 0, width, height);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

		display_func();
		display_func();
		f_mandelbrot->fractal_render(0);
		display_func();
	}
}

void initialize_GL()
{
	glfwSetErrorCallback(print_error);
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
	glfwSetMouseButtonCallback(window, mouse_click);
	glfwSetCursorPosCallback(window, mouse_motion);
	glfwSetWindowSizeCallback(window, resize);
	glfwSetScrollCallback(window, mouse_scroll);

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
	printf("\nRenderer: \t\t%s\n", renderer);
	printf("OpenGL Version: \t%s\n", version);
}

void initialize_config()
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
		if (token.compare("ResolutionX") == 0)
		{
			config.resolutionX = std::stoi(field);
		}
		if (token.compare("ResolutionY") == 0)
		{
			config.resolutionY = std::stoi(field);
		}

		if (token.compare("InstructionSet") == 0)
		{
			config.instructionSet = std::stoi(field);
		}
		if (token.compare("Threads") == 0)
		{
			config.threadMax = std::stoi(field);
		}
		if (token.compare("SSAA") == 0)
		{
			config.SSAA = std::stoi(field);
		}
		if (token.compare("ShadingMode") == 0)
		{
			config.shadingMode = std::stoi(field);
		}
		if (token.compare("SmoothShading") == 0)
		{
			config.smoothShading = std::stoi(field);
		}
		if (token.compare("ExecutionMode") == 0)
		{
			config.executionMode = std::stoi(field);
		}
		if (token.compare("PaletteFile") == 0)
		{
			config.paletteFile = field;
		}

		if (token.compare("Zoom_Coarse") == 0)
		{
			config.zoom_coarse = std::stod(field);
		}
		if (token.compare("Zoom_Fine") == 0)
		{
			config.zoom_fine = std::stod(field);
		}
		if (token.compare("Zoom_Wheel") == 0)
		{
			config.zoom_wheel = std::stod(field);
		}

		if (token.compare("Iteration_Max_Red") == 0)
		{
			config.iteration_max_red = std::stoul(field);
		}

		if (token.compare("Iteration_Max_Green") == 0)
		{
			config.iteration_max_green = std::stoul(field);
		}

		if (token.compare("Iteration_Max_Blue") == 0)
		{
			config.iteration_max_blue = std::stoul(field);
		}

		if (token.compare("Sample_Size_Red") == 0)
		{
			config.sample_size_red = std::stoul(field);
		}

		if (token.compare("Sample_Size_Green") == 0)
		{
			config.sample_size_green = std::stoul(field);
		}

		if (token.compare("Sample_Size_Blue") == 0)
		{
			config.sample_size_blue = std::stoul(field);
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

		config.palette[4 * paletteIndex + 3] = 255;
		config.palette[4 * paletteIndex + 2] = (unsigned char)paletteRed;
		config.palette[4 * paletteIndex + 1] = (unsigned char)paletteGreen;
		config.palette[4 * paletteIndex + 0] = (unsigned char)paletteBlue;

		std::cout << "Palette " << paletteIndex << ": " << (int)config.palette[4 * paletteIndex + 0] << " " << (int)config.palette[4 * paletteIndex + 1] << " " << (int)config.palette[4 * paletteIndex + 2] << std::endl;
	}
	palette.close();

	//modify the palette so that colors do not immediately switch at the boundary
	config.palette[4 * (config.paletteNumber) + 3] = config.palette[3];
	config.palette[4 * (config.paletteNumber) + 2] = config.palette[2];
	config.palette[4 * (config.paletteNumber) + 1] = config.palette[1];
	config.palette[4 * (config.paletteNumber) + 0] = config.palette[0];

	config.xmin = (config.xcoord - ((double)config.resolutionX / config.resolutionY)*(config.zoomFactor));
	config.xmax = (config.xcoord + ((double)config.resolutionX / config.resolutionY)*(config.zoomFactor));
	config.ymin = (config.ycoord - (config.zoomFactor));
	config.ymax = (config.ycoord + (config.zoomFactor));

	config.centerX = (config.xmax + config.xmin) / 2;
	config.centerY = (config.ymax + config.ymin) / 2;
	config.windowX = config.xmax - config.xmin;
	config.windowY = config.ymax - config.ymin;

	config.bufferX = config.resolutionX * config.SSAA;
	config.bufferY = config.resolutionY * config.SSAA;
}

void render_loop()
{
	initialize_GL();
	initialize_buffers(config.resolutionX, config.resolutionY, true);
	do // Main Loop
	{
		glfwPollEvents();
		command_prompt_poll();
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	} while (!glfwWindowShouldClose(window));
	glfwDestroyWindow(window);
	glfwTerminate();
	delete_buffers();
}

////////////////////////////////////////
// Command Handling

std::vector<std::string> vectorize_input()
{
	std::vector<std::string> tokens;
	std::ifstream input;
	std::string line;
	std::string token;
	char iterator;

	std::getline(std::cin, line);
	line += '\n';
	while (!line.empty())
	{
		iterator = line.at(0);
		if (iterator == '/')
		{
			iterator = line.at(1);
			if (iterator == '/')
			{
				break; // Skip Line
			}
			else
			{
				iterator = line.at(0);
			}
		}

		if (iterator == '{' || iterator == '}' || iterator == '[' || iterator == ']')
		{
			if (token.length() > 0)
			{
				// Commit current input token
				tokens.push_back(token);
				token = "";
				// Commit reserved symbol
				tokens.push_back(line.substr(0, 1));
				line.erase(line.begin());
			}
			else
			{
				tokens.push_back(line.substr(0, 1));
				line.erase(line.begin());
			}
		}
		else if (!isspace(iterator))
		{
			// Grab next symbol
			token.append(line.substr(0, 1));
			line.erase(line.begin());
		}
		else if (token.length() > 0 && (isspace(iterator)))
		{
			// Commit current input token
			tokens.push_back(token);
			token = "";
			line.erase(line.begin());
		}
		else
		{
			line.erase(line.begin());
		}
	}

	std::reverse(tokens.begin(), tokens.end());

	return tokens;
}

void command_prompt_enqueue(int type)
{
	Command *comm;
	if (type == 0 || type == 3)
	{
		comm = new Command(config.centerX, config.centerY, config.zoomFactor);
		comm->type = type;
	}
	else if (type == 1)
	{
		comm = new Command(config.maxIter);
		comm->type = type;
	}
	else if (type == 2)
	{
		comm = new Command(config.centerX, config.centerY, config.zoomFactor);
		comm->type = type;
	}
	else if (type == 4)
	{
		comm = new Command(config.centerX, config.centerY, config.zoomFactor);
		comm->type = type;
	}
	else if (type == 5)
	{
		comm = new Command(config.centerX, config.centerY, config.zoomFactor);
		comm->type = type;
	}
	commandQueue.push_back(comm);
}

void command_prompt_poll()
{
	if (commandQueue.empty() == false)
	{
		Command *comm = commandQueue.back();
		commandQueue.pop_back();
		std::thread render;
		if (comm->type == 2)
		{
			config.bufferX = config.resolutionX * config.SSAA;
			config.bufferY = config.resolutionY * config.SSAA;
			initialize_buffers(config.resolutionX, config.resolutionY, true);
			
			f_mandelbrot->fractal_render(1);
			display_func();
		}
		else if (comm->type == 3) // Buddhabrot
		{
			f_buddhabrot->fractal_render(1);
			display_func();
		}
		else if (comm->type == 4) // Clear Buffers
		{
			delete_buffers();
		}
		else if (comm->type == 5) // Init
		{
			initialize_buffers(config.resolutionX, config.resolutionY, false);
		}
		else
		{
			f_mandelbrot->fractal_render(1);
			display_func();
		}
	}
}

void display_help()
{
	printf("\nOpenGL Renderer Commands\n");
	printf("[left mouse] Change center\n");
	printf("[mouse wheel up] Zoom in : %fx\n", config.zoom_wheel);
	printf("[mouse wheel down] Zoom out : %fx\n\n", config.zoom_wheel);

	printf("Course Adjustment\n");
	printf("[w] Zoom in : %fx\n", config.zoom_coarse);
	printf("[s] Zoom out : %fx\n", config.zoom_coarse);
	printf("[a] Increase iterations : 2x\n");
	printf("[d] Decrease iterations : 2x\n\n");

	printf("Fine Adjustment\n");
	printf("CTRL + [w] Zoom in : %fx\n", config.zoom_fine);
	printf("CTRL + [s] Zoom out : %fx\n", config.zoom_fine);
	printf("CTRL + [a] Increase iterations : 1\n");
	printf("CTRL + [d] Decrease iterations : 1\n\n");

	printf("[q] Auto iterate : 256\n");
	printf("[/] Change Instruction Set\n");
	printf("[.] Toggle Render Optimization\n\n");
	printf("[?] This Display\n\n");

	printf("Terminal Mode Commands\n");
	printf("\"point x y r\" \t refocus the graph onto a specified point.\n");
	printf("\"resize x y\" \t resize the resolution of the window.\n");
	printf("\"ssaa n\" \t change supersampling level, set to 1 to disable.\n");
}

int main(int argc, char **argv)
{
	initialize_config();

	std::thread renderWindow;
	renderWindow = std::thread(render_loop);

	f_mandelbrot = new F_Mandelbrot(calculation, config);

	std::ifstream options;
	std::string line;
	std::string command;
	std::string field;

	display_help();
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
			double x, y, r;

			std::stringstream stream(field);
			stream >> x;
			stream.ignore();
			stream >> y;
			stream.ignore();
			stream >> r;
			stream.ignore();

			action_point(x, y, r);
		}
		else if (command.compare("resize") == 0)
		{
			int x = NULL, y = NULL;

			std::stringstream stream(field);
			stream >> x;
			stream.ignore();
			stream >> y;
			stream.ignore();

			if (x == NULL && y == NULL)
			{
				printf("resolution x = %d, y = %d\n", config.resolutionX, config.resolutionY);
			}
			else if (x == NULL || x < 0)
			{
				printf("x param bad\n");
			}
			else if (y == NULL && x > 0)
			{
				glfwSetWindowSize(window, x, x);
			}
			else if (y == NULL && x < 0)
			{
				printf("x param bad\n");
			}
			else
			{
				glfwSetWindowSize(window, x, y);
			}
		}
		else if (command.compare("ssaa") == 0)
		{
			int n = NULL;
			std::stringstream stream(field);
			stream >> n;
			stream.ignore();

			if (n == NULL)
			{
				printf("ssaa = %d\n", config.SSAA);
			}
			else if (n < 0)
			{
				printf("ssaa param bad\n");
			}
			else
			{
				action_ssaa(n);
			}
		}
		else if (command.compare("save_image") == 0)
		{
			std::string filename;
			std::stringstream stream(field);
			stream >> filename;
			if (filename.compare("") == 0)
			{
				printf("filename is empty\n");
			}
			else
			{
				filename += ".bmp";
				output_bitmap(filename);
				printf("Image saved as %s\n", filename.c_str());
			}
		}
		else if (command.compare("save_render") == 0)
		{
			std::stringstream stream(field);
			std::string filename, fractal;
			int x = NULL, y = NULL;
			stream >> filename;
			stream >> fractal;
			stream >> x;
			stream >> y;

			if (filename.compare("") == 0)
			{
				printf("filename is empty\n");
			}
			else if (fractal.compare("") == 0)
			{
				printf("fractal is empty\n");
			}
			else if (fractal.compare("mandelbrot") != 0
				&& fractal.compare("buddhabrot") != 0)
			{
				printf("fractal unrecognized\n");
			}
			else if (x == NULL || x < 0)
			{
				printf("x param bad\n");
			}
			else if (y == NULL && x < 0)
			{
				printf("x param bad\n");
			}
			else if (y == NULL && x > 0)
			{
				y = x;
				filename += ".bmp";
				printf("Rendering %s with resolution [%d,%d]\n", fractal.c_str(), x, y);
				output_render(filename, fractal, x, y);
				printf("Image saved as %s\n", filename.c_str());
			}
			else
			{
				filename += ".bmp";
				printf("Rendering %s with resolution [%d,%d]\n", fractal.c_str(), x, y);
				output_render(filename, fractal, x, y);
				printf("Image saved as %s\n", filename.c_str());
			}
		}
		else if (command.compare("buddhabrot") == 0)
		{
			action_buddha();
		}
		else if (command.compare("exit") == 0)
		{
			glfwDestroyWindow(window);
			glfwTerminate();
			delete_buffers();
			return 0;
		}
		else
		{
			printf("Unrecognized Command\n");
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	} while (true);

	renderWindow.join();
	exit(EXIT_SUCCESS);

	return 0;
}
////////////////////////////////////////
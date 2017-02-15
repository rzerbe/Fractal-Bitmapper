#include "Command.h"



Command::Command()
{
}


Command::~Command()
{
}

Command::Command(double x_new, double y_new, double r_new)
{
	this->x_new = x_new;
	this->y_new = y_new;
	this->r_new = r_new;

	this->comm_type_screen = 1;
}
Command::Command(int iter_new)
{
	this->iter_new = iter_new;

	this->comm_type_iter = 1;
}

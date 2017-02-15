#pragma once
class Command
{
public:
	Command();
	~Command();

	Command(double x_new, double y_new, double r_new);
	Command(int iter_new);
	int type;
private:
	bool comm_type_screen;
	bool comm_type_iter;
	double x_new;
	double y_new;
	double r_new;
	int iter_new;
};


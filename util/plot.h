#pragma once

#include <string>
#include <vector>

struct Scatter{
    std::string line_name;
    std::vector<double> x;
    std::vector<double> y;
};

struct PlotArgs{
    std::string out_path;
    std::string graph_name;
    std::string x_axis_title;
    std::string y_axis_title;
    std::vector<Scatter> data;
};

void Plot(const PlotArgs& graph);
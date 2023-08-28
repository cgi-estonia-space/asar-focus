#include "plot.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#include <boost/algorithm/string.hpp>

namespace {
const char* HTML_TEMPLATE = R"foo(
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <script src="https://cdn.plot.ly/plotly-2.25.2.min.js" charset="utf-8"></script>
    <script charset="utf-8">

    </script>
</head>
<body>
        <div id="gd"></div>

    <script>

var data = [$$0$$];

Plotly.newPlot('gd', data, $$1$$);

    </script>
</body>
</html>
)foo";
}
void Plot(const PlotArgs& graph) {
    std::string base_html(HTML_TEMPLATE);
    std::stringstream data;

    for (int i = 0; i < graph.data.size(); i++) {
        auto& line = graph.data[i];
        data << "{x:[";
        for (int j = 0; j < line.x.size(); j++) {
            data << line.x[j];
            if (j + 1 != line.x.size()) {
                data << ",";
            }
        }
        data << "],y:[";
        for (int j = 0; j < line.y.size(); j++) {
            data << line.y[j];
            if (j + 1 != line.y.size()) {
                data << ",";
            }
        }
        data << "],type:'scatter',name:'";
        data << line.line_name;
        data << "'}";
        if (i + 1 != graph.data.size()) {
            data << ",";
        }
    }

    boost::replace_first(base_html, "$$0$$", data.str());

    std::stringstream layout;
    layout << "{title:'" << graph.graph_name << "',xaxis:{title:'" << graph.x_axis_title << "'},yaxis:{title:'"
           << graph.y_axis_title << "'}}";

    boost::replace_first(base_html, "$$1$$", layout.str());

    std::ofstream ofs(graph.out_path);
    ofs << base_html;
    printf("Output plot @ %s\n", graph.out_path.c_str());
}

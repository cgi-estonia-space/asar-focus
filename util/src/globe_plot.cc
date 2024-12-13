/*
 * ENVISAT and ERS ASAR instrument focusser for QA4EO activity (c)
 * by CGI Estonia AS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "globe_plot.h"

#include <fstream>

#include <fmt/format.h>

#include "alus_log.h"
#include "geo_tools.h"
#include "sar/orbit_state_vector.h"

namespace {
const char* HTML_TEMPLATE = R"foo(
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="utf-8">


    <script src="https://cesium.com/downloads/cesiumjs/releases/1.112/Build/Cesium/Cesium.js"></script>

    <link href="https://cesium.com/downloads/cesiumjs/releases/1.112/Build/Cesium/Widgets/widgets.css" rel="stylesheet">

</head>

<style>
  table {

  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

td{
padding: 6px 12px;
}
tr{
  background: #f6f6f6;
}
tr:nth-of-type(odd) {
  background: #e9e9e9;
}
tr:nth-of-type(1) {
  font-weight: 900;
  color: #ffffff;
  background: #27ae60;
}
</style>

<body>

  <div>
    <a href="https://cesium.com/learn/ion/cesium-ion-access-tokens/">API key</a><input type="text" id="api_key"/><br>
    Range IDX<input type="range" id="rg_input"/><br>
    Azimuth IDX<input type="range" id="az_input"/><br>
    Display zero Doppler plane?<input type="checkbox" id="display_plane" checked="true"/><br>
    Display range bubble?<input type="checkbox" id="display_bubble" checked="true"/>
    </div>
    <div id="result">
      </div>

    <div id="cesiumContainer"></div>

    <table id="sar_info">
      <tr><th>variable</th><th>value</th></tr>
    </table>

    <table id="orbits"><tr><th>time</th><th>X pos</th><th>Y pos</th><th>Y pos</th><th>X vel</th><th>Y vel</th><th>Z vel</th></tr></table>


    <script type="module">


// @#$ GENERATED DATA BEGIN @#$
{{{REPLACE}}}
// @#$ GENERATED DATA END @#$



    //UI variables

    const rg_input = document.getElementById("rg_input");
    const az_input = document.getElementById("az_input");
    const api_key_input = document.getElementById("api_key");

    let viewer = null;

    function init_cesium_viewer()
    {
      if(viewer !== null)
      {
        return;
      }
      Cesium.Ion.defaultAccessToken = api_key_input.value;
      viewer = new Cesium.Viewer('cesiumContainer', {
        terrain: Cesium.Terrain.fromWorldTerrain(),
      });

      viewer.animation.container.style.visibility = "hidden";
      viewer.timeline.container.style.visibility = "hidden";
      viewer.forceResize();

      let corner_points =  metadata.corner_UL.concat(metadata.corner_UR, metadata.corner_LR, metadata.corner_LL);
      viewer.entities.add({
        polygon: {
          hierarchy: Cesium.Cartesian3.fromDegreesArray(corner_points),
          height: 0,
          material: Cesium.Color.RED.withAlpha(0.5),
          outline: true,
          outlineColor: Cesium.Color.BLACK,
        },
      });

      for (let o of osv) {
        viewer.entities.add({
          position: new Cesium.Cartesian3.fromArray(o.pos),
          point: {
            pixelSize: 10,
            color: Cesium.Color.BLUE,
          }
        });
      }
      api_key_input.remove();
      inputs_changed();
    }

    function datestr_to_ms(date_str)
    {
        let dt = date_str.slice(0, -7);
        let us_str = date_str.slice(-6);
        let ms = parseFloat(us_str)/1000.0;
        let unix_ms = Date.parse(dt);
        let tot_ms = unix_ms + ms;
        return tot_ms;
    }

    function interpolate_orbit(time_ms)
    {

      let ret = {
        time_ms: time_ms,
        x_pos: 0,
        x_vel: 0,
        y_pos: 0,
        y_vel: 0,
        z_pos: 0,
        z_vel: 0,
      };

      for (let i = 0; i < osv.length; i++) {
        let mult = 1;
        for (let j = 0; j < osv.length; j++) {
          if (i === j) { continue; }
          mult *=
            (time_ms - osv[j].time_ms) / (osv[i].time_ms - osv[j].time_ms);

        }
        ret.x_pos += mult * osv[i].pos[0];
        ret.y_pos += mult * osv[i].pos[1];
        ret.z_pos += mult * osv[i].pos[2];
        ret.x_vel += mult * osv[i].vel[0];
        ret.y_vel += mult * osv[i].vel[1];
        ret.z_vel += mult * osv[i].vel[2];
      }

      return ret;
    }

    function init_data_display()
    {
      {
        let t = document.getElementById("sar_info");
        let i = 0;
        for(let [key, value] of Object.entries(metadata))
        {
            let r = t.insertRow(i + 1);
            r.insertCell(0).innerHTML = key;
            if(value.constructor === Array){
                value = value.toString().split(",").join(", ")
            }
            r.insertCell(1).innerHTML = value;
            i++;
        }
      }

      {
        let t = document.getElementById("orbits");
        for(let i = 0; i < osv.length; i++)
        {
          let r = t.insertRow(i + 1);
          r.insertCell(0).innerHTML = osv[i].time;
          r.insertCell(1).innerHTML = osv[i].pos[0].toFixed(3);
          r.insertCell(2).innerHTML = osv[i].pos[1].toFixed(3);
          r.insertCell(3).innerHTML = osv[i].pos[2].toFixed(3);
          r.insertCell(4).innerHTML = osv[i].vel[0].toFixed(3);
          r.insertCell(5).innerHTML = osv[i].vel[1].toFixed(3);
          r.insertCell(6).innerHTML = osv[i].vel[2].toFixed(3);
        }
      }

      let rg_i = document.getElementById("rg_input");
      rg_input.min = 0;
      rg_input.value = 0;
      rg_input.max = metadata.range_size - 1;
      az_input.min = 0;
      az_input.value = 0;
      az_input.max = metadata.az_size - 1;

      for (let o of osv) {
          o.time_ms = datestr_to_ms(o.time);
      }

      api_key_input.value = "";
      rg_input.addEventListener("change", inputs_changed);
      az_input.addEventListener("change", inputs_changed);
      api_key_input.addEventListener("change", init_cesium_viewer);
    }

    var range_bubble_cs = null;
    var sat_pos_cs = null;
    var zero_doppler_cs = null;

    function inputs_changed() {
      if(viewer === null) {
        return;
      }
      const rg_idx = rg_input.value;
      const az_idx = az_input.value;

      let first = datestr_to_ms(metadata.first_line_time);
      let lti_ms = (1/metadata.prf) * 1000;
      const interp_time = first + az_idx * lti_ms;

      const interp = interpolate_orbit(interp_time);

      let sat_vel = new Cesium.Cartesian3(interp.x_vel, interp.y_vel, interp.z_vel);
      let sat_pos = new Cesium.Cartesian3(interp.x_pos, interp.y_pos, interp.z_pos);

      // interpolated orbit position
      if(sat_pos_cs === null) {
        sat_pos_cs = viewer.entities.add({
          position: sat_pos,
          point: {
            pixelSize: 10,
            color: Cesium.Color.CYAN
          }
        });
      }
      sat_pos_cs.position = sat_pos;

      //range bubble
      const radius = metadata.slant_range_first + rg_idx * metadata.range_spacing;
      if(range_bubble_cs === null) {
        range_bubble_cs = viewer.entities.add({
          position: sat_pos,
          ellipsoid: {
            radii: new Cesium.Cartesian3(1, 1, 1),
            material: new Cesium.Color(1.0, 1.0, 1.0, 0.25),
          },
        });
      }
      range_bubble_cs.position = sat_pos;
      range_bubble_cs.ellipsoid.radii = new Cesium.Cartesian3(radius, radius, radius);
      range_bubble_cs.show = document.getElementById("display_bubble").checked;

      // zero doppler plane

      let norm_vel = new Cesium.Cartesian3();
      Cesium.Cartesian3.normalize(sat_vel, norm_vel);

      if(zero_doppler_cs === null) {
        zero_doppler_cs = viewer.entities.add({
        position: new Cesium.Cartesian3(0, 0, 0),
        orientation: Cesium.Quaternion.IDENTITY,
        plane: {
          plane: new Cesium.Plane(new Cesium.Cartesian3(1, 0, 0), 0.0),
          dimensions: new Cesium.Cartesian2(10000000, 10000000),
          material: new Cesium.Color(1, 1, 0, 0.5),
        }
        }
        );
      }

      zero_doppler_cs.position = sat_pos;
      zero_doppler_cs.plane.plane = new Cesium.Plane(norm_vel, 0.0);
      zero_doppler_cs.show = document.getElementById("display_plane").checked;

      // display results
      let r_div = document.getElementById("result");
      let rt = "";
      rt = "";
      rt += "<table><tr><th>Name</th><th>Calculated value</th><th>Comment</th></tr>";
      rt += "<tr><td>Range</td><td>" + rg_idx.toString() + "</td></tr>";
      rt += "<tr><td>Azimuth</td><td>" + az_idx.toString() + "</td></tr>";
      rt += "<tr><td>Slant range</td><td>" + radius.toFixed(3) + "</td><td>Radius of the white bubble</td></tr>";
      rt += "<tr><td>Time</td><td>" + new Date(interp_time).toISOString() + "</td></tr>";
      rt += "<tr><td>Satellite position</td><td>" + sat_pos.toString() + "</td><td>satellite position(cyan point)</td></tr>";
      rt += "<tr><td>Satellite velocity</td><td>" + sat_vel.toString() + "</td><td>normal vector to the zero doppler plane(yellow)</tr>";
      rt += "<tr><td>Num orbit state vectors</td><td>" + osv.length.toString() + "</td><td>Interpolation data points(blue)</tr>";
      rt += "</table>";

      r_div.innerHTML = rt;
    }

    function init_cesium_display()
    {
      let corner_points =  metadata.corner_UL.concat(metadata.corner_UR, metadata.corner_LR, metadata.corner_LL);
      viewer.entities.add({
        polygon: {
          hierarchy: Cesium.Cartesian3.fromDegreesArray(corner_points),
          height: 0,
          material: Cesium.Color.RED.withAlpha(0.5),
          outline: true,
          outlineColor: Cesium.Color.BLACK,
        },
      });


      for (let o of osv) {
        viewer.entities.add({
          position: new Cesium.Cartesian3.fromArray(o.pos),
          point: {
            pixelSize: 10,
            color: Cesium.Color.BLUE,
          },
        });
      }
    }

    function calc_range_bubble_radius(range_idx)
    {
        return metadata.slant_range_first + range_idx * metadata.range_spacing;
    }

    window.onload = function ()
    {
      init_data_display();
    };
    </script>
    </div>
</body>
</html>
)foo";

// TODO copied from sar folder, make sar a lib in the future to avoid duplication
OrbitStateVector InterpolateOrbitCopy(const std::vector<OrbitStateVector>& osv, boost::posix_time::ptime time) {
    double time_point = (time - osv.front().time).total_microseconds() * 1e-6;

    OrbitStateVector r = {};
    r.time = time;
    const int n = osv.size();
    for (int i = 0; i < n; i++) {
        double mult = 1;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            double xj = (osv.at(j).time - osv.front().time).total_microseconds() * 1e-6;
            double xi = (osv.at(i).time - osv.front().time).total_microseconds() * 1e-6;
            mult *= (time_point - xj) / (xi - xj);
        }

        r.x_pos += mult * osv[i].x_pos;
        r.y_pos += mult * osv[i].y_pos;
        r.z_pos += mult * osv[i].z_pos;
        r.x_vel += mult * osv[i].x_vel;
        r.y_vel += mult * osv[i].y_vel;
        r.z_vel += mult * osv[i].z_vel;
    }

    return r;
}
}  // namespace

// Generate a html, which visualizes geolocation and sar metadata on a globe
// the HTML_TEMPLATE variable contains the necessary user interface code
// C++ code inserts SAR metadata as javascript variables
void PlotGlobe(const SARMetadata& sar_meta, const std::string& path) {
    double slant_range_first = CalcSlantRange(sar_meta, 0);
    double slant_range_last = CalcSlantRange(sar_meta, sar_meta.img.range_size - 1);
    int rg_size = sar_meta.img.range_size;
    int az_size = sar_meta.img.azimuth_size;
    auto start_time = CalcAzimuthTime(sar_meta, 0);
    auto last_line_time = CalcAzimuthTime(sar_meta, az_size - 1);
    GeoPosLLH ul_llh;
    GeoPosLLH ur_llh;
    GeoPosLLH ll_llh;
    GeoPosLLH lr_llh;

    {
        auto o = InterpolateOrbitCopy(sar_meta.osv, start_time);
        auto p = RangeDopplerGeoLocate({o.x_vel, o.y_vel, o.z_vel}, {o.x_pos, o.y_pos, o.z_pos}, sar_meta.center_point,
                                       slant_range_first);
        ul_llh = xyz2geoWGS84(p);
    }

    {
        auto o = InterpolateOrbitCopy(sar_meta.osv, start_time);
        auto p = RangeDopplerGeoLocate({o.x_vel, o.y_vel, o.z_vel}, {o.x_pos, o.y_pos, o.z_pos}, sar_meta.center_point,
                                       slant_range_last);
        ur_llh = xyz2geoWGS84(p);
    }

    {
        auto o = InterpolateOrbitCopy(sar_meta.osv, last_line_time);
        auto p = RangeDopplerGeoLocate({o.x_vel, o.y_vel, o.z_vel}, {o.x_pos, o.y_pos, o.z_pos}, sar_meta.center_point,
                                       slant_range_first);
        ll_llh = xyz2geoWGS84(p);
    }

    {
        auto o = InterpolateOrbitCopy(sar_meta.osv, last_line_time);
        auto p = RangeDopplerGeoLocate({o.x_vel, o.y_vel, o.z_vel}, {o.x_pos, o.y_pos, o.z_pos}, sar_meta.center_point,
                                       slant_range_last);
        lr_llh = xyz2geoWGS84(p);
    }

    std::string s;
    s += fmt::format("const metadata = {{\n");
    s += fmt::format("\"range_size\" : {},\n", rg_size);
    s += fmt::format("\"az_size\" : {},\n", az_size);
    s += fmt::format("\"range_spacing\" : {},\n", sar_meta.range_spacing);
    s += fmt::format("\"slant_range_first\" : {},\n", sar_meta.slant_range_first_sample);
    {
        auto time_str = to_iso_extended_string(sar_meta.first_line_time);
        if ((sar_meta.first_line_time.time_of_day().total_microseconds() % 1000000) == 0) {
            time_str += ".000000";
        }
        s += fmt::format("\"first_line_time\" : \"{}\",\n", time_str);
    }
    s += fmt::format("\"prf\":{},\n", sar_meta.pulse_repetition_frequency);
    s += fmt::format("\"corner_UL\" : [{},{}],\n", ul_llh.longitude, ul_llh.latitude);
    s += fmt::format("\"corner_UR\" : [{},{}],\n", ur_llh.longitude, ur_llh.latitude);
    s += fmt::format("\"corner_LL\" : [{},{}],\n", ll_llh.longitude, ll_llh.latitude);
    s += fmt::format("\"corner_LR\" : [{},{}],\n", lr_llh.longitude, lr_llh.latitude);
    s += fmt::format("}};\n");
    s += fmt::format("const osv = [");
    for (size_t i = 0; i < sar_meta.osv.size(); i++) {
        const auto& osv = sar_meta.osv[i];

        auto t = osv.time;
        auto time_str = to_iso_extended_string(t);
        if ((t.time_of_day().total_microseconds() % 1000000) == 0) {
            time_str += ".000000";  // boost does not add microseconds, if it is 0
        }
        s += fmt::format("{{\n\"time\": \"{}\",\n", time_str);
        s += fmt::format("\"pos\": [{}, {}, {}],\n", osv.x_pos, osv.y_pos, osv.z_pos);
        s += fmt::format("\"vel\": [{}, {}, {}],\n", osv.x_vel, osv.y_vel, osv.z_vel);
        s += fmt::format("}}{}\n", i == sar_meta.osv.size() - 1 ? "" : ",");
    }

    s += fmt::format("];\n");

    std::string html = HTML_TEMPLATE;

    const std::string magic = "{{{REPLACE}}}";
    html.replace(html.find(magic), magic.size(), s);

    std::ofstream ofs(path);
    ofs << html;

    LOGD << "Cesium globe plot @ " << path;
}
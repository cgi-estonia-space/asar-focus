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

#include "img_output.h"

#include <queue>
#include <thread>

#include <gdal/gdal_priv.h>

#include "alus_log.h"
#include "device_padded_image.cuh"
#include "math_utils.h"

namespace {

// Optimized file write with GDAL datasets, which are stored as strips as opposed to tiles
// 1) Uses WriteBlock API, to bypass GDAL Cache

void WriteTiff(const DevicePaddedImage& img, bool complex_output, bool padding, const char* path, bool cut) {
    (void)cut;
    const int w = padding ? img.XStride() : img.XSize();
    const int h = padding ? img.YStride() : img.YSize();

    LOGI << "Writing " << (complex_output ? "complex" : "intensity") << " file at " << path;

    const int buf_size = w * h;

    const auto gdal_type = complex_output ? GDT_CFloat32 : GDT_Float32;

    char** output_driver_options = nullptr;
    GDALAllRegister();
    auto drv = GetGDALDriverManager()->GetDriverByName("gtiff");
    auto* ds = drv->Create(path, w, h, 1, gdal_type, output_driver_options);

    std::unique_ptr<GDALDataset, decltype(&GDALClose)> gdal_close(ds, GDALClose);

    auto* band = ds->GetRasterBand(1);

    int x_block_size;
    int y_block_size;
    band->GetBlockSize(&x_block_size, &y_block_size);

    LOGV << "Write slow path";
    std::unique_ptr<cufftComplex[]> data(new cufftComplex[buf_size]);
    if (padding) {
        img.CopyToHostPaddedSize(data.get());
    } else {
        img.CopyToHostLogicalSize(data.get());
    }

    auto* b = ds->GetRasterBand(1);
    if (complex_output) {
        const auto res = b->RasterIO(GF_Write, 0, 0, w, h, data.get(), w, h, GDT_CFloat32, 0, 0);
        (void)res; // TODO - Check the result.
    } else {
        // convert to intensity, each pixel being I^2 + Q^2
        std::unique_ptr<float[]> intens_buf(new float[buf_size]);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = x + y * w;

                float i = data[idx].x;
                float q = data[idx].y;
                intens_buf[idx] = i * i + q * q;
            }
        }
        const auto res = b->RasterIO(GF_Write, 0, 0, w, h, intens_buf.get(), w, h, GDT_Float32, 0, 0);
        (void)res; // TODO - Check the result.
    }
}

}  // namespace

void WriteIntensityPaddedImg(const DevicePaddedImage& img, const char* path) {
    WriteTiff(img, false, false, path, false);
}

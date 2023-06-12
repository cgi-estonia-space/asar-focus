/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */

#include "img_output.h"

#include <condition_variable>
#include <queue>
#include <thread>

#include <gdal/gdal_priv.h>
#include <boost/date_time.hpp>

//#include "alus_log.h"
#include "checks.h"
//#include "gdal_util.h"
#include "math_utils.h"
//#include "pugixml.hpp"

// 0 -> Tiff
// 1 -> Envi
// 2 -> BE Envi via hack, no direct gdal support?

#define OUTPUT_FORMAT 0

namespace {

    float Bswapf32(float i) {
        uint32_t t;
        memcpy(&t, &i, 4);
        t = __builtin_bswap32(t);
        memcpy(&i, &t, 4);
        return i;
    }

    struct ImgSlice {
        cufftComplex *slice_data;
        int slice_cnt;
        int slice_start_idx;
        int slice_stride;
    };

    struct SharedData {
        // gpu thread moves data from device to host and gives exclusive access for that buffer for file IO
        struct {
            std::queue<ImgSlice> queue;
            std::condition_variable cv;
            std::mutex mutex;
            bool stop_file_write;
        } gpu;

        // file thread consumes the memory for file IO and tells the gpu threads the memory can be reused for cudaMemcpy
        struct {
            std::queue<cufftComplex *> queue;
            std::condition_variable cv;
            std::mutex mutex;
            CPLErr gdal_err;
        } file;
    };

    void RowWriterTask(GDALRasterBand *band, SharedData *shared) {
        int rows_written = 0;
        const int total_rows = band->GetYSize();
        while (rows_written < total_rows) {
            ImgSlice el;
            {
                std::unique_lock l(shared->gpu.mutex);
                shared->gpu.cv.wait(l, [=]() { return shared->gpu.stop_file_write || !shared->gpu.queue.empty(); });
                if (shared->gpu.stop_file_write) {
                    return;
                }
                el = shared->gpu.queue.front();
                shared->gpu.queue.pop();
            }

            // write blocks from
            for (int i = 0; i < el.slice_cnt; i++) {
                int y_block = el.slice_start_idx + i;
                auto *ptr = el.slice_data + i * el.slice_stride;
                auto e = band->WriteBlock(0, y_block, ptr);
                if (e == CE_None) {
                    rows_written++;
                } else {
                    {
                        std::unique_lock l(shared->file.mutex);
                        shared->file.gdal_err = e;
                    }
                    shared->file.cv.notify_one();
                    return;
                }
            }

            {
                // release memory back for reuse
                std::unique_lock l(shared->file.mutex);
                shared->file.queue.push(el.slice_data);
            }
            //printf("slice done = %d %d\n", el.slice_start_idx, el.slice_cnt);
            shared->file.cv.notify_one();
        }
    }



    // Optimized file write with GDAL datasets, which are stored as strips as opposed to tiles
// 1) Uses WriteBlock API, to bypass GDAL Cache
// 2) Run cudaMemcpy(& complex -> intensity transform) in parallel to file writing
    void WriteStripedDataset(const DevicePaddedImage& img, GDALRasterBand* band, bool complex_output, bool y_pad, bool cut) {
        constexpr size_t QUEUE_SIZE = 3;
        SharedData sd = {};
        std::array<std::unique_ptr<cufftComplex[]>, QUEUE_SIZE> ptr_array;

        const int n_rows_per_slice = 10;
        const int x_stride = img.XStride();
        const int y_size = y_pad ? img.YStride() : img.YSize();

        int aperature_size = 3000;

        for (auto& el : ptr_array) {
            // TODO(priit) benchmark if pinned memory be worth if the file thread writes to ramdisk?
            el = std::unique_ptr<cufftComplex[]>(new cufftComplex[x_stride * n_rows_per_slice]);
            sd.file.queue.push(el.get());
        }

        std::thread gdal_thread(RowWriterTask, band, &sd);

        int rows_transferred = 0;
        cudaError cuda_err = {};

        while (rows_transferred < y_size) {
            cufftComplex* h_ptr = nullptr;
            {
                std::unique_lock l(sd.file.mutex);
                sd.file.cv.wait(l, [&]() { return sd.file.gdal_err != CE_None || !sd.file.queue.empty(); });

                if (sd.file.gdal_err != CE_None) {
                    break;
                }
                h_ptr = sd.file.queue.front();
                sd.file.queue.pop();
            }

            const auto* d_src = img.Data() + rows_transferred * x_stride;

            int act_rows = n_rows_per_slice;
            if (rows_transferred + n_rows_per_slice > y_size) {
                act_rows = y_size - rows_transferred;
            }
            ImgSlice slice = {};
            cuda_err = cudaMemcpy(h_ptr, d_src, act_rows * x_stride * 8, cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess) {
                {
                    std::unique_lock l(sd.gpu.mutex);
                    sd.gpu.stop_file_write = true;
                }
                sd.gpu.cv.notify_one();
                break;
            }

            if (cut) {
                for (int i = 0; i < act_rows; i++) {
                    auto* ptr = h_ptr + (i * x_stride);
                    const int y = rows_transferred + i;
                    if (y < aperature_size || y + aperature_size > y_size) {
                        memset(ptr, 0, sizeof(ptr[0]) * x_stride);
                        continue;
                    }
                }
            }
            if (!complex_output) {
                const int x_size = band->GetXSize();
                for (int i = 0; i < act_rows; i++) {
                    InplaceComplexToIntensity(h_ptr + (i * x_stride), x_size);
                }
            }

            slice.slice_data = h_ptr;
            slice.slice_stride = x_stride;
            slice.slice_start_idx = rows_transferred;
            slice.slice_cnt = act_rows;
            //printf("slice = %d %d add\n", rows_transferred, act_rows);
            {
                std::unique_lock l(sd.gpu.mutex);
                sd.gpu.queue.push(slice);
            }
            sd.gpu.cv.notify_one();
            rows_transferred += act_rows;
        }
        if(0)
        {
            {
                std::unique_lock l(sd.gpu.mutex);
                sd.gpu.stop_file_write = true;
            }
            sd.gpu.cv.notify_one();
        }
        gdal_thread.join();

        CHECK_CUDA_ERR(cuda_err);
        CHECK_GDAL_ERROR(sd.file.gdal_err);
    }
// Optimized file write with GDAL datasets, which are stored as strips as opposed to tiles
// 1) Uses WriteBlock API, to bypass GDAL Cache

    void WriteTiff(const DevicePaddedImage &img, bool complex_output, bool padding, const char *path, bool cut) {
        const int w = padding ? img.XStride() : img.XSize();
        const int h = padding ? img.YStride() : img.YSize();
#if OUTPUT_FORMAT == 1 || OUTPUT_FORMAT == 2
        std::string wo_ext = path;

    auto idx = wo_ext.find(".tif");
    if (idx < wo_ext.size()) {
        wo_ext.resize(idx);
        LOGD << "Envi name = " << wo_ext;
        path = wo_ext.c_str();
    }
    if (!complex_output) {
        complex_output = true;
        LOGD << "COMPLEX OUTPUT OVERRIDE";
    }
#endif

        std::cout << "Writing " << (complex_output ? "complex" : "intensity") << " file @ " << path << "\n";

        const int buf_size = w * h;

        const auto gdal_type = complex_output ? GDT_CFloat32 : GDT_Float32;

        char **output_driver_options = nullptr;
        GDALAllRegister();
        auto drv = GetGDALDriverManager()->GetDriverByName("gtiff");
#if OUTPUT_FORMAT == 0
        //auto *drv = alus::GetGdalGeoTiffDriver();
#elif OUTPUT_FORMAT == 1 || OUTPUT_FORMAT == 2
        auto* drv = GetGDALDriverManager()->GetDriverByName("ENVI");
#endif
        auto *ds = drv->Create(path, w, h, 3, gdal_type, output_driver_options);


        std::unique_ptr<GDALDataset, decltype(&GDALClose)> gdal_close(ds, GDALClose);

        auto *band = ds->GetRasterBand(1);

        int x_block_size;
        int y_block_size;
        band->GetBlockSize(&x_block_size, &y_block_size);
        if (x_block_size == w && y_block_size == 1 && false) {
            WriteStripedDataset(img, band, complex_output, padding, true);
            return;
        }


        std::cout << "Write slow path\n";
        std::unique_ptr<cufftComplex[]> data(new cufftComplex[buf_size]);
        if (padding) {
            img.CopyToHostPaddedSize(data.get());
        } else {
            img.CopyToHostLogicalSize(data.get());
        }

        auto *b = ds->GetRasterBand(1);
        auto* b_i = ds->GetRasterBand(2);
        auto* b_q = ds->GetRasterBand(3);
        if (complex_output) {
#if OUTPUT_FORMAT == 2
            for (int i = 0; i < w * h; i++) {
            data[i].x = Bswapf32(data[i].x);
            data[i].y = Bswapf32(data[i].y);
        }
#endif
            b->RasterIO(GF_Write, 0, 0, w, h, data.get(), w, h, GDT_CFloat32, 0, 0);
        } else {
            // convert to intensity, each pixel being I^2 + Q^2
            std::unique_ptr<float[]> i_buf(new float[buf_size]);
            std::unique_ptr<float[]> q_buf(new float[buf_size]);
            std::unique_ptr<float[]> intens_buf(new float[buf_size]);
            for(int y = 0; y < h; y++)
            {
                for(int x = 0; x < w; x++)
                {
                    int idx = x + y*w;
                    if(y < 3000 || y > (h - 3000))
                    {
                        intens_buf[idx] = 0;
                        i_buf[idx] = 0;
                        q_buf[idx] = 0;
                    }
                    else
                    {
                        float i = data[idx].x;
                        float q = data[idx].y;
                        intens_buf[idx] = i*i + q*q;
                        i_buf[idx] = i;
                        q_buf[idx] = q;
                    }
                }
            }
            b->RasterIO(GF_Write, 0, 0, w, h, intens_buf.get(), w, h, GDT_Float32, 0, 0);
            b_i->RasterIO(GF_Write, 0, 0, w, h, i_buf.get(), w, h, GDT_Float32, 0, 0);
            b_q->RasterIO(GF_Write, 0, 0, w, h, q_buf.get(), w, h, GDT_Float32, 0, 0);

        }
    }

}  // namespace



void WriteIntensityPaddedImg(const DevicePaddedImage& img, const char* path, bool cut)
{
    WriteTiff(img, false, false, path, cut);
}



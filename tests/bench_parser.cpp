// Host-side sanity benchmark (NOT a LOCI measurement, NOT a Jetson number).
//
// Feeds a full-size YOLOv7 640x640 output tensor (25,200 rows x 85 floats) with
// a realistic score distribution through NvDsInferParseCustomYoloV7 and reports
// the wall-clock time per call on whatever CPU runs it. Its only purpose is to
// show that the handful-of-boxes unit test cannot see what a real frame costs:
// run it on the original parser, on the innocent edit and on the fix and compare.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseCustomYoloV7(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static const int NUM_CLASSES = 80;
static const int ROW_WIDTH = 85;
static const int ROWS = 25200;  // 3 heads x 3 anchors x (80*80 + 40*40 + 20*20)

static unsigned s_seed = 12345u;
static float frand() { s_seed = s_seed * 1103515245u + 12345u; return (float)((s_seed >> 8) & 0xffffff) / 16777216.0f; }

int main(int argc, char** argv)
{
    const int iters = argc > 1 ? std::atoi(argv[1]) : 3;
    std::vector<float> tensor((size_t)ROWS * ROW_WIDTH);
    // Boxes spread over the 640x640 frame, 24-64 px wide; objectness mostly
    // low (background) with ~2% confident rows, like a real frame.
    for (int r = 0; r < ROWS; ++r) {
        float* row = &tensor[(size_t)r * ROW_WIDTH];
        row[0] = 32.f + frand() * 576.f;
        row[1] = 32.f + frand() * 576.f;
        row[2] = 24.f + frand() * 40.f;
        row[3] = 24.f + frand() * 40.f;
        const bool confident = frand() < 0.02f;
        row[4] = confident ? 0.6f + frand() * 0.4f : frand() * 0.2f;
        for (int c = 0; c < NUM_CLASSES; ++c) row[5 + c] = frand() * 0.3f;
        row[5 + (r % NUM_CLASSES)] = confident ? 0.7f + frand() * 0.3f : 0.3f + frand() * 0.3f;
    }

    NvDsInferLayerInfo layer;
    std::memset(&layer, 0, sizeof(layer));
    layer.dataType = FLOAT;
    layer.inferDims.numDims = 2;
    layer.inferDims.d[0] = ROWS;
    layer.inferDims.d[1] = ROW_WIDTH;
    layer.inferDims.numElements = ROWS * ROW_WIDTH;
    layer.layerName = "output";
    layer.buffer = tensor.data();
    std::vector<NvDsInferLayerInfo> layers(1, layer);

    NvDsInferNetworkInfo net; net.width = 640; net.height = 640; net.channels = 3;
    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = NUM_CLASSES;
    params.perClassPreclusterThreshold.assign(NUM_CLASSES, 0.25f);
    params.perClassPostclusterThreshold.assign(NUM_CLASSES, 0.25f);

    double best_ms = 1e300;
    size_t nout = 0;
    for (int i = 0; i < iters; ++i) {
        std::vector<NvDsInferParseObjectInfo> out;
        auto t0 = std::chrono::steady_clock::now();
        NvDsInferParseCustomYoloV7(layers, net, params, out);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best_ms) best_ms = ms;
        nout = out.size();
    }
    std::printf("bench_parser: %d rows -> %zu boxes, best of %d: %.3f ms per call (host wall clock, not a Jetson or LOCI figure)\n",
                ROWS, nout, iters, best_ms);
    return 0;
}

// Live-demo shim (host only, NOT what LOCI measures).
//
// Linked into one dylib per parser variant (upstream, innocent edit, fix) next
// to that variant's nvdsparsebbox_Yolo.cpp, unchanged. It builds the same
// NvDsInferLayerInfo / NvDsInferParseDetectionParams that nvinfer hands the
// custom parser, calls NvDsInferParseCustomYoloV7 on one real model output
// tensor, times just that call, and copies the objects out as flat floats for
// the Python front end.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseCustomYoloV7(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static const int NUM_CLASSES = 80;

// out: maxOut rows of {left, top, width, height, confidence, classId}.
// Returns the number of rows written (-1 if the parser returned false);
// *totalOut is the parser's full object count, *parseMs its wall-clock time.
extern "C" int demo_parse(const float* tensor, int rows, int dims, int netW, int netH,
                          float preclusterThreshold, float* out, int maxOut,
                          int* totalOut, double* parseMs)
{
    NvDsInferLayerInfo layer;
    std::memset(&layer, 0, sizeof(layer));
    layer.dataType = FLOAT;
    layer.inferDims.numDims = 2;
    layer.inferDims.d[0] = rows;
    layer.inferDims.d[1] = dims;
    layer.inferDims.numElements = rows * dims;
    layer.layerName = "output";
    layer.buffer = const_cast<float*>(tensor);
    std::vector<NvDsInferLayerInfo> layers(1, layer);

    NvDsInferNetworkInfo net;
    net.width = netW;
    net.height = netH;
    net.channels = 3;

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = NUM_CLASSES;
    params.perClassPreclusterThreshold.assign(NUM_CLASSES, preclusterThreshold);
    params.perClassPostclusterThreshold.assign(NUM_CLASSES, preclusterThreshold);

    std::vector<NvDsInferParseObjectInfo> objects;
    auto t0 = std::chrono::steady_clock::now();
    const bool ok = NvDsInferParseCustomYoloV7(layers, net, params, objects);
    auto t1 = std::chrono::steady_clock::now();
    *parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    *totalOut = (int)objects.size();
    if (!ok) return -1;

    const int n = std::min((int)objects.size(), maxOut);
    for (int i = 0; i < n; ++i) {
        const NvDsInferParseObjectInfo& o = objects[i];
        float* r = out + (size_t)i * 6;
        r[0] = o.left;
        r[1] = o.top;
        r[2] = o.width;
        r[3] = o.height;
        r[4] = o.detectionConfidence;
        r[5] = (float)o.classId;
    }
    return n;
}

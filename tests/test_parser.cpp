// Unit test for NVIDIA's DeepStream YOLO CPU parser (nvdsparsebbox_Yolo.cpp).
//
// Feeds a synthetic YOLOv7 output tensor of a handful of rows (85 floats each:
// x, y, w, h, objectness, 80 class scores) and a synthetic YOLOv4 pair of
// tensors through the exported parse entry points, and checks that every
// confident box comes back with the right class and geometry and that a
// degenerate (zero-width) box is dropped.
//
// The test is deliberately the kind a reviewer accepts: it checks that the
// boxes we care about are present and correct. It does not feed a full
// 25,200-row tensor and it does not time anything, so it passes on the
// original parser, on the innocent edit (NMS over all rows) and on the fix
// alike. That is the point of the demo: the tests say go.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseCustomYoloV4(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

extern "C" bool NvDsInferParseCustomYoloV7(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static int g_failures = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            ++g_failures;                                                  \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);      \
            std::fprintf(stderr, __VA_ARGS__);                             \
            std::fprintf(stderr, "\n");                                    \
        }                                                                  \
    } while (0)

static const int NUM_CLASSES = 80;
static const int ROW_WIDTH = 5 + NUM_CLASSES;  // 85
static const unsigned NET_W = 640, NET_H = 640;

struct Expect {
    unsigned classId;
    float left, top, width, height;
};

static bool near(float a, float b, float tol = 0.51f) { return std::fabs(a - b) <= tol; }

static bool contains(const std::vector<NvDsInferParseObjectInfo>& out, const Expect& e)
{
    for (size_t i = 0; i < out.size(); ++i) {
        const NvDsInferParseObjectInfo& o = out[i];
        if (o.classId == e.classId && near(o.left, e.left) && near(o.top, e.top) &&
            near(o.width, e.width) && near(o.height, e.height))
            return true;
    }
    return false;
}

static NvDsInferParseDetectionParams makeParams(float preThreshold)
{
    NvDsInferParseDetectionParams p;
    p.numClassesConfigured = NUM_CLASSES;
    p.perClassPreclusterThreshold.assign(NUM_CLASSES, preThreshold);
    p.perClassPostclusterThreshold.assign(NUM_CLASSES, preThreshold);
    return p;
}

static NvDsInferLayerInfo makeLayer(const char* name, float* buffer,
                                    unsigned d0, unsigned d1, unsigned d2, unsigned numDims)
{
    NvDsInferLayerInfo l;
    std::memset(&l, 0, sizeof(l));
    l.dataType = FLOAT;
    l.inferDims.numDims = numDims;
    l.inferDims.d[0] = d0;
    l.inferDims.d[1] = d1;
    l.inferDims.d[2] = d2;
    l.inferDims.numElements = d0 * (numDims > 1 ? d1 : 1) * (numDims > 2 ? d2 : 1);
    l.bindingIndex = 0;
    l.layerName = name;
    l.buffer = buffer;
    l.isInput = 0;
    return l;
}

// One YOLOv7 row: centre x/y, w/h in network pixels, objectness, one class score.
static void putRowV7(std::vector<float>& t, int row, float cx, float cy, float w, float h,
                     float objectness, int cls, float score)
{
    float* r = &t[row * ROW_WIDTH];
    r[0] = cx; r[1] = cy; r[2] = w; r[3] = h; r[4] = objectness;
    for (int c = 0; c < NUM_CLASSES; ++c) r[5 + c] = 0.0f;
    r[5 + cls] = score;
}

static void testYoloV7()
{
    const int rows = 5;
    std::vector<float> tensor(rows * ROW_WIDTH, 0.0f);
    //                 row  cx     cy     w     h     obj    cls score
    putRowV7(tensor,   0,   125.f, 125.f, 50.f, 50.f, 0.90f, 0,  0.95f);  // person
    putRowV7(tensor,   1,   340.f, 220.f, 80.f, 40.f, 0.80f, 5,  0.90f);  // bus
    putRowV7(tensor,   2,   530.f, 430.f, 60.f, 60.f, 0.70f, 79, 0.80f);  // toothbrush
    putRowV7(tensor,   3,   100.f, 500.f, 30.f, 30.f, 0.05f, 3,  0.50f);  // weak row
    putRowV7(tensor,   4,   300.f, 300.f, 0.0f, 20.f, 0.99f, 1,  0.99f);  // zero width

    std::vector<NvDsInferLayerInfo> layers;
    layers.push_back(makeLayer("output", tensor.data(), rows, ROW_WIDTH, 0, 2));

    NvDsInferNetworkInfo net;
    net.width = NET_W; net.height = NET_H; net.channels = 3;
    NvDsInferParseDetectionParams params = makeParams(0.25f);

    std::vector<NvDsInferParseObjectInfo> out;
    bool ok = NvDsInferParseCustomYoloV7(layers, net, params, out);
    CHECK(ok, "NvDsInferParseCustomYoloV7 returned false");

    const Expect want[] = {
        {0,  100.f, 100.f, 50.f, 50.f},
        {5,  300.f, 200.f, 80.f, 40.f},
        {79, 500.f, 400.f, 60.f, 60.f},
    };
    for (size_t i = 0; i < sizeof(want) / sizeof(want[0]); ++i) {
        CHECK(contains(out, want[i]),
              "YOLOv7: expected box class=%u left=%.0f top=%.0f w=%.0f h=%.0f missing",
              want[i].classId, want[i].left, want[i].top, want[i].width, want[i].height);
    }
    for (size_t i = 0; i < out.size(); ++i) {
        CHECK(out[i].width >= 1.0f && out[i].height >= 1.0f,
              "YOLOv7: degenerate box (w=%.2f h=%.2f) was not dropped", out[i].width, out[i].height);
        CHECK(out[i].classId < (unsigned)NUM_CLASSES, "YOLOv7: classId %u out of range", out[i].classId);
        CHECK(out[i].left >= 0.f && out[i].top >= 0.f && out[i].left + out[i].width <= NET_W + 0.5f &&
              out[i].top + out[i].height <= NET_H + 0.5f, "YOLOv7: box outside the network frame");
    }
    std::printf("YOLOv7: %zu rows in, %zu boxes out\n", (size_t)rows, out.size());
}

static void testYoloV4()
{
    const int n = 3;
    std::vector<float> boxes(n * 4, 0.0f);     // normalised x1,y1,x2,y2
    std::vector<float> scores(n * NUM_CLASSES, 0.0f);
    const float bx[n][4] = {
        {0.10f, 0.10f, 0.20f, 0.20f},
        {0.50f, 0.25f, 0.75f, 0.50f},
        {0.80f, 0.80f, 0.90f, 0.95f},
    };
    const int cls[n] = {2, 7, 41};
    const float sc[n] = {0.9f, 0.8f, 0.7f};
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 4; ++k) boxes[i * 4 + k] = bx[i][k];
        scores[i * NUM_CLASSES + cls[i]] = sc[i];
    }

    std::vector<NvDsInferLayerInfo> layers;
    layers.push_back(makeLayer("boxes", boxes.data(), n, 1, 4, 3));
    layers.push_back(makeLayer("scores", scores.data(), n, NUM_CLASSES, 0, 2));

    NvDsInferNetworkInfo net;
    net.width = NET_W; net.height = NET_H; net.channels = 3;
    NvDsInferParseDetectionParams params = makeParams(0.25f);

    std::vector<NvDsInferParseObjectInfo> out;
    bool ok = NvDsInferParseCustomYoloV4(layers, net, params, out);
    CHECK(ok, "NvDsInferParseCustomYoloV4 returned false");
    CHECK(out.size() == (size_t)n, "YOLOv4: expected %d boxes, got %zu", n, out.size());
    for (int i = 0; i < n; ++i) {
        Expect e = {(unsigned)cls[i], bx[i][0] * NET_W, bx[i][1] * NET_H,
                    (bx[i][2] - bx[i][0]) * NET_W, (bx[i][3] - bx[i][1]) * NET_H};
        CHECK(contains(out, e), "YOLOv4: expected box class=%u left=%.0f top=%.0f missing",
              e.classId, e.left, e.top);
    }
    std::printf("YOLOv4: %d rows in, %zu boxes out\n", n, out.size());
}

int main()
{
    testYoloV7();
    testYoloV4();
    if (g_failures) {
        std::printf("test_parser: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_parser: all checks passed\n");
    return 0;
}

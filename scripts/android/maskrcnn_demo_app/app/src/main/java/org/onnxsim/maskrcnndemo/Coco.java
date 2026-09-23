package org.onnxsim.maskrcnndemo;

/**
 * Label names for the ONNX model zoo MaskRCNN-12 (maskrcnn-benchmark lineage): 81 contiguous
 * classes, background + the 80 COCO categories in order (cat = 16, remote = 66). Not torchvision's
 * 91-id list with "N/A" gaps.
 */
final class Coco {
    static final String[] NAMES = {
        "__background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
        "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"};

    static String name(int id) {
        return id >= 0 && id < NAMES.length ? NAMES[id] : ("id" + id);
    }
}

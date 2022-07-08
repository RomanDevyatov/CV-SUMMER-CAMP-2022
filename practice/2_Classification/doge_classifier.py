"""
Classification sample

Command line to run:
python doge_classifier.py -i image.jpg \
    -m mobilenet-v2-pytorch.xml -c imagenet_synset_words.txt
"""

import os
import cv2
import sys
import argparse
import numpy as np
import logging as log
from openvino.runtime import Core


class InferenceEngineClassifier:

    def __init__(self, model_path, device='CPU', classes_path=None):
        
        # Add code for Inference Engine initialization
        self.core = Core()
        
        # Add code for model loading
        self.model = self.core.read_model(model=model_path)

        self.exec_model = self.core.compile_model(model=self.model,
                                                  device_name=device)

        # Add code for classes names loading
        self.classes = None
        if classes_path:
            self.classes = [line.rstrip('\n') for line in open(classes_path)]


    def get_top(self, prob, topN=1):
        result = []
        
        # Add code for getting top predictions
        array = np.array(prob)

        args = np.argsort(array[0, :])[-topN:]

        for arg in args:
            result.append(str(self.classes[arg]) + " prob: " + str(array[0, :][arg]))

        return result

    def _prepare_image(self, image, h, w):
    
        # Add code for image preprocessing
        image = cv2.resize(image, (w, h))
        # rgbrgbrgb to rrrgggbbb
        image = image.transpose((2, 0, 1))

        # get [image amount, color channel amount
        image = np.expand_dims(image, axis=0)
        
        return image

    def classify(self, image):
        # Add code for image classification using Inference Engine
        input_layer = self.exec_model.input(0)
        output_layer = self.exec_model.output(0)

        n, c, h, w = input_layer.shape

        prepared_image = self._prepare_image(image, h, w)

        # request = self.exec_model.create_infer_request()
        # request.infer(inputs={input_layer.any_name: prepared_image})
        # result = request.get_output_tensor(output_layer.index).data

        result = self.exec_model([prepared_image])[output_layer]

        return result


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
        image file', required=True, type=str)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-c', '--classes', help='File containing classes \
        names', type=str, default=None)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Start IE classification sample")

    # Create InferenceEngineClassifier object
    ie_classifier = InferenceEngineClassifier(model_path=args.model,
                                              classes_path=args.classes)
    
    # Read image
    image = cv2.imread(args.input)
        
    # Classify image
    prob = ie_classifier.classify(image)
    
    # Get top 5 predictions
    predictions = ie_classifier.get_top(prob, 5)
    
    # print result
    log.info("Predictions: \n" + "\n".join(predictions))


if __name__ == '__main__':
    sys.exit(main())

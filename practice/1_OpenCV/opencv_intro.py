import argparse
import sys

import cv2


def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    original_image = cv2.imread(input_image_path)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray_image)

    # Normalize image intensity
    normalize_image = cv2.normalize(gray_image, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX)
    cv2.imshow("normalize_image", normalize_image)

    # Resize image
    processed_image = cv2.resize(normalize_image, (640, 480), interpolation=cv2.INTER_AREA)
    cv2.imshow("processed_image", processed_image)

    # Detect cat faces using Haar Cascade
    cascade = cv2.CascadeClassifier(haar_model_path)

    processed_image_copy = processed_image.copy()
    cat_rects = cascade.detectMultiScale(processed_image_copy,
                                                scaleFactor=1.2,
                                                minNeighbors=5)

    # Draw bounding box
    for (x, y, w, h) in cat_rects:
        cv2.rectangle(processed_image, (x, y),
                      (x + w, y + h), (255, 255, 255), 10)
    cv2.imshow("Processed image", processed_image)

    # Crop image
    x, y, w, h = cat_rects[0]
    croped_original_image = original_image[y:y + h, x:x + w]

    # Display result image
    cv2.imshow("Cropped", croped_original_image)

    # Save result image to file
    cv2.imwrite("croped_cat_face.jpg", croped_original_image)

    # Put cropeed image on passport image
    passport_image = cv2.imread("pet_passport.png")
    
    x_offset = y_offset = 45
    passport_image[y_offset:y_offset + croped_original_image.shape[0], x_offset:x_offset + croped_original_image.shape[1]] = croped_original_image

    # Put text on passport
    cv2.putText(img=passport_image, text='Tom', org=(90, 218), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                color=(0, 100, 200), thickness=2)
    cv2.putText(img=passport_image, text='Male', org=(100, 258), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                color=(0, 100, 200), thickness=2)
    cv2.putText(img=passport_image, text='6396fg30s', org=(330, 84), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                color=(0, 100, 200), thickness=2)

    # Save passport photo
    cv2.imshow("Final passport", passport_image)
    cv2.imwrite("processed_passport.jpg", passport_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def build_argparser():
    parser = argparse.ArgumentParser(description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to .XML file with pre-trained model.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')

    return parser


def main():
    args = build_argparser().parse_args()
    make_cat_passport_image(args.input, args.model)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)

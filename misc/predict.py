import os
import tempfile
from pathlib import Path
import glob
import numpy as np
import cv2
from zipfile import ZipFile
from PIL import Image
import shutil
import cog
from face_enhancement import FaceEnhancement
from face_colorization import FaceColorization
from face_inpainting import FaceInpainting, brush_stroke_mask


class Predictor(cog.Predictor):
    def setup(self):
        faceenhancer_model = {'name': 'GPEN-BFR-256', 'size': 256, 'channel_multiplier': 1, 'narrow': 0.5}
        self.faceenhancer = FaceEnhancement(size=faceenhancer_model['size'], model=faceenhancer_model['name'],
                                            channel_multiplier=faceenhancer_model['channel_multiplier'],
                                            narrow=faceenhancer_model['narrow'])
        faceinpainter_model = {'name': 'GPEN-Inpainting-1024', 'size': 1024}
        self.faceinpainter = FaceInpainting(size=faceinpainter_model['size'], model=faceinpainter_model['name'],
                                            channel_multiplier=2)
        facecolorizer_model = {'name': 'GPEN-Colorization-1024', 'size': 1024}
        self.facecolorizer = FaceColorization(size=facecolorizer_model['size'], model=facecolorizer_model['name'],
                                              channel_multiplier=2)

    @cog.input(
        "image",
        type=Path,
        help="input image",
    )
    @cog.input(
        "task",
        type=str,
        options=['Face Restoration', 'Face Colorization', 'Face Inpainting'],
        default='Face Restoration',
        help="choose task type"
    )
    @cog.input(
        "output_individual",
        type=bool,
        default=False,
        help="whether outputs individual enhanced faces, valid for Face Restoration. When set to true, a zip folder of "
             "all the enhanced faces in the input will be generated for download."
    )
    @cog.input(
        "broken_image",
        type=bool,
        default=True,
        help="whether the input image is broken, valid for Face Inpainting. When set to True, the output will be the "
             "'fixed' image. When set to False, the image will randomly add brush strokes to simulate a broken image, "
             "and the output will be broken + fixed image"
    )
    def predict(self, image, task='Face Restoration', output_individual=False, broken_image=True):
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        if task == 'Face Restoration':
            im = cv2.imread(str(image), cv2.IMREAD_COLOR)  # BGR
            assert isinstance(im, np.ndarray), 'input filename error'
            im = cv2.resize(im, (0, 0), fx=2, fy=2)
            img, orig_faces, enhanced_faces = self.faceenhancer.process(im)
            cv2.imwrite(str(out_path), img)
            if output_individual:
                zip_folder = 'out_zip'
                os.makedirs(zip_folder, exist_ok=True)
                out_path = Path(tempfile.mkdtemp()) / "out.zip"
                try:
                    cv2.imwrite(os.path.join(zip_folder, 'whole_image.jpg'), img)
                    for m, ef in enumerate(enhanced_faces):
                        cv2.imwrite(os.path.join(zip_folder, f'face_{m}.jpg'), ef)
                    img_list = sorted(glob.glob(os.path.join(zip_folder, '*')))
                    with ZipFile(str(out_path), 'w') as zipfile:
                        for img in img_list:
                            zipfile.write(img)
                finally:
                    clean_folder(zip_folder)
        elif task == 'Face Colorization':
            grayf = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            grayf = cv2.cvtColor(grayf, cv2.COLOR_GRAY2BGR)  # channel: 1->3
            colorf = self.facecolorizer.process(grayf)
            cv2.imwrite(str(out_path), colorf)
        else:
            originf = cv2.imread(str(image), cv2.IMREAD_COLOR)
            brokenf = originf
            if not broken_image:
                brokenf = np.asarray(brush_stroke_mask(Image.fromarray(originf)))
            completef = self.faceinpainter.process(brokenf)
            brokenf = cv2.resize(brokenf, completef.shape[:2])
            out_img = completef if broken_image else np.hstack((brokenf, completef))
            cv2.imwrite(str(out_path), out_img)

        return out_path


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
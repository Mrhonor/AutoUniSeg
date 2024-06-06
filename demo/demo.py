import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from auto_uni_seg import add_hrnet_config, add_gnn_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)

    add_hrnet_config(cfg)
    add_gnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/7_datasets/vis.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            # ds = ['cs', 'mp', 'sun', 'bdd', 'idd', 'ade', 'co']
            ds = ['cs', 'mp', 'sun', 'bdd', 'idd', 'ade', 'co', 'cv', 'voc', 'ct', 'sn', 'kt']
            # ds = ['ade', 'bdd', 'mp']
            cur_name = 'uni'
            for d in ds:
                list1 = os.listdir(os.path.join(path, d))
                for l in list1:
                    if 'png' not in l and 'jpg' not in l:
                        continue
                    out_dir = os.path.join(path, d)
                    img = read_image(os.path.join(out_dir, l), format="BGR")
                    start_time = time.time()
                    # predictions, visualized_output, visual_uni_output = demo.run_on_image(img)
                    predictions, visual_uni_output = demo.run_on_image(img)
                    logger.info(
                        "{}: {} in {:.2f}s".format(
                            path,
                            "detected {} instances".format(len(predictions["instances"]))
                            if "instances" in predictions
                            else "finished",
                            time.time() - start_time,
                        )
                    )
                    
                    out_dir = os.path.join(out_dir, 'out')
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    out_dir = os.path.join(out_dir, l)
                    # visual_uni_output.save(out_dir.replace('.png', '_uni.png').replace('.jpg', '_uni.png'))
                    out_dir = out_dir.replace('.jpg', f'_{cur_name}.jpg').replace('.png', f'_{cur_name}.jpg')
                    visual_uni_output.save(out_dir)
                    im = cv2.imread(out_dir)
                    im = cv2.resize(im, (1120, 840))
                    cv2.imwrite(out_dir, im)
                    # for vis in visualized_output

            # if args.output:
            #     if os.path.isdir(args.output):
            #         assert os.path.isdir(args.output), args.output
            #         out_filename = os.path.join(args.output, os.path.basename(path))
            #     else:
            #         assert len(args.input) == 1, "Please specify a directory with args.output"
            #         out_filename = args.output
            #     ds_name = ['cs', 'mp', 'sun', 'bdd', 'idd', 'ade', 'co']
            #     # for i in range(len(visualized_output)):
            #     #     visualized_output[i].save(out_filename.replace('.png', f'_{ds_name[i]}.png').replace('.jpg', f'_{ds_name[i]}.jpg'))
            #     visual_uni_output.save(out_filename.replace('.png', '_uni.png').replace('.jpg', '_uni.jpg'))
            # else:
            #     cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            #     cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            #     if cv2.waitKey(0) == 27:
            #         break  # esc to quit

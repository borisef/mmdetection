import copy
import glob
import os.path
import os.path as osp
import sys

import cv2
import mmcv
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from train_wrap_utils_home import replace_test_in_cfg
from tools import test
from tools.analysis_tools import confusion_matrix
from tools.analysis_tools import test_robustness


TEST_ROBUSTNESS = False
to_show = True





test_imgs_dir = "/home/borisef/data/vehicles/test/" #None
test_annotations = "/home/borisef/data/vehicles/test/annotations_5classes.json"#None

config_file = "/home/borisef/projects/mm/mmdetection/boris3/config/config_faster-rcnn_r50_fpn_2x_coco.py"
ckpt_path = "/home/borisef/projects/mm/mmdetection/boris3/work_dirs/faster-rcnn_r50_fpn_2x_vehicles/epoch_24.pth"

out_dir = "/home/borisef/temp/out_test/"

thresh = 0.5

if(not os.path.exists(out_dir)):
    os.mkdir(out_dir)

if(test_imgs_dir is not None):
    temp_cnfig = os.path.join(out_dir, "temp_cnfg.py")
    replace_test_in_cfg(config_file,temp_cnfig,test_imgs_dir,test_annotations)
    config_file = temp_cnfig




sys.argv.append(config_file)
backup_sys_argv = sys.argv.copy()

sys.argv.append(ckpt_path)
sys.argv.append("--out")
sys.argv.append(osp.join(out_dir,'results.pickle'))
if(to_show):
    sys.argv.append("--show")
    sys.argv.append("--wait-time")
    sys.argv.append("0.1")
sys.argv.append("--show-dir")
sys.argv.append(out_dir)
#--show-dir
# --wait-time

### TEST ###
test.main()
# # Single-gpu testing
# python tools/test.py \
#     ${CONFIG_FILE} \
#     ${CHECKPOINT_FILE} \
#     [--out ${RESULT_FILE}] \
#     [--show]



### CONF MAT ####

sys.argv = backup_sys_argv.copy()
sys.argv.append(osp.join(out_dir,'results.pickle'))
sys.argv.append(out_dir)
if(to_show):
    sys.argv.append("--show")



confusion_matrix.main() #${CONFIG}  ${DETECTION_RESULTS}  ${SAVE_DIR} --show

if(TEST_ROBUSTNESS):
    #### CORRUPTIONS
    sys.argv = backup_sys_argv.copy()
    sys.argv.append(ckpt_path)
    sys.argv.append("--out")
    sys.argv.append(osp.join(out_dir,'results_corrupt.pickle'))
    sys.argv.append("--corruptions")
    sys.argv.append("noise")

    # python tools/analysis_tools/test_robustness.py ${CONFIG_FILE}
    # ${CHECKPOINT_FILE}
    # [--out ${RESULT_FILE}]
    test_robustness.main()
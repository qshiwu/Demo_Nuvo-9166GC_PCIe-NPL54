import cv2
import time
import argparse
import os
import torch

import posenet
from posenet.decode_multi import *

parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=int, default=101)
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=0.5)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    filenames = [
        f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    start = time.time()

     # Open video capture from /dev/video102
    cap = cv2.VideoCapture('/dev/video102')

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        

        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)


        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
            pose_scores, keypoint_scores, keypoint_coords, pose_offsets = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)
        
        keypoint_coords *= output_scale
        
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                # cv2.circle(frame, (250,250), 10, (255,255,255), 3)
                # if isinstance(c[0], (int, float)) and isinstance(c[1], (int, float)):                    
                cv2.circle(frame, (int(round(c[0])), int(round(c[1]))) , 10, (255,0,0), 1)

                


        cv2.imshow("POSENET", frame)
        # frame_with_skeleton = posenet.draw_keypoints()


        

        ## 

        # Wait for key press
        key = cv2.waitKey(3) & 0xFF

        # Wait for Q or ESC
        if key in [27, ord('q'), ord('Q')]:
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    print('Demo Completed')
    # for f in filenames:
    #     input_image, draw_image, output_scale = posenet.read_imgfile(
    #         f, scale_factor=args.scale_factor, output_stride=output_stride)

    #     with torch.no_grad():
    #         input_image = torch.Tensor(input_image).cuda()

    #         heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
    #         pose_scores, keypoint_scores, keypoint_coords, pose_offsets = posenet.decode_multi.decode_multiple_poses(
    #             heatmaps_result.squeeze(0),
    #             offsets_result.squeeze(0),
    #             displacement_fwd_result.squeeze(0),
    #             displacement_bwd_result.squeeze(0),
    #             output_stride=output_stride,
    #             max_pose_detections=10,
    #             min_pose_score=0.25)

    #     keypoint_coords *= output_scale

    #     if args.output_dir:
    #         draw_image = posenet.draw_skel_and_kp(
    #             draw_image, pose_scores, keypoint_scores, keypoint_coords,
    #             min_pose_score=0.25, min_part_score=0.25)

    #         cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

    #     if not args.notxt:
    #         print()
    #         print("Results for image: %s" % f)
    #         for pi in range(len(pose_scores)):
    #             if pose_scores[pi] == 0.:
    #                 break
    #             print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
    #             for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
    #                 print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

    # print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()

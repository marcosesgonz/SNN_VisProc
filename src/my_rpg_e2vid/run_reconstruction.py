import torch
import os
from utils.loading_utils import myload_model, get_device
import numpy as np
import argparse
from utils.event_readers import FixedSizeEventReader, MyFixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
import time
from image_reconstructor import myImageReconstructor
from options.inference_options import set_inference_options


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    parser.add_argument('-i', '--input_file', required=True, type=str)
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=False)
    parser.add_argument('--save_images_in_png', dest='save_images_in_png', action='store_true')
    parser.set_defaults(save_images_in_png=False)
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--time_steps', default = 22, type = float,
                        help="Number of event windows with a fixed duration. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    set_inference_options(parser)

    args = parser.parse_args()

    path_to_events = args.input_file

    data_name_root = path_to_events
    for n in range(4):
        data_name_root = os.path.dirname(data_name_root)

    data_name = os.path.basename(data_name_root)
    if data_name in ['DVS_Gesture_dataset','DVS_Animals_Dataset','DVS_DailyAction_dataset']:
        width, height = 128, 128
    elif data_name == 'DVS_ActionRecog_dataset':
        width, height = 346, 260

    # Load model
    model = myload_model(args.path_to_model)
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()

    reconstructor = myImageReconstructor(model, height, width, model.num_bins,device, args)


    # Loop through the events and reconstruct images
    N = args.window_size
    if not args.fixed_duration:
        if N is None:
            N = int(width * height * args.num_events_per_pixel)
            print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
                N, args.num_events_per_pixel))
        else:
            print('Will use {} events per tensor (user-specified)'.format(N))
            mean_num_events_per_pixel = float(N) / float(width * height)
            if mean_num_events_per_pixel < 0.1:
                print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))
            elif mean_num_events_per_pixel > 1.5:
                print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset

    if args.compute_voxel_grid_on_cpu:
        print('Will compute voxel grid on CPU.')

    if args.fixed_duration:
        event_window_iterator = MyFixedDurationEventReader(path_to_events,
                                                         time_steps=args.time_steps)
                                                        #start_index=start_index)
    else:
        raise NotImplementedError('Only fixed duration reconstruction implemented')

    frames = []
    for event_window in event_window_iterator:

        last_timestamp = event_window[-1, 0]

        if args.compute_voxel_grid_on_cpu:
            event_tensor = events_to_voxel_grid(event_window,
                                                num_bins=model.num_bins,
                                                width=width,
                                                height=height)
            event_tensor = torch.from_numpy(event_tensor)
        else:
            event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                        num_bins=model.num_bins,
                                                        width=width,
                                                        height=height,
                                                        device=device)

        num_events_in_window = event_window.shape[0]
        new_frame = reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window,
                                                last_timestamp, save_png = args.save_images_in_png)

        start_index += num_events_in_window
        frames.append(new_frame)

    fname = os.path.join(args.output_folder, os.path.basename(path_to_events))
    np.savez(fname,video = frames)
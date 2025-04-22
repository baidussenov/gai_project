import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# Configuration
dataset_root = "~/Desktop/gai-project/BVI-Lowlight-videos"  # Update to your dataset path
output_root = "./output_masks"  # Where to save mask images and arrays
save_masked_frames = False  # Set to True to save masked frames
num_scenes = 40  # Number of scenes to process (e.g., 20 or 40)

# Global variables for rectangle selection
drawing = False
start_point = None
end_point = None
mask_rect = None
temp_rect = None  # To store rectangle during drawing

def clamp_mask_to_frame(mask, frame_height, frame_width, mask_size):
    """Adjust mask position to stay fully within frame, preserving scene-specific mask size."""
    fixed_width, fixed_height = mask_size
    x1, y1 = mask[0], mask[1]
    x2 = x1 + fixed_width
    y2 = y1 + fixed_height
    if x2 > frame_width:
        x1 = frame_width - fixed_width
        x2 = frame_width
    if y2 > frame_height:
        y1 = frame_height - fixed_height
        y2 = frame_height
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = x1 + fixed_width
    y2 = y1 + fixed_height
    clamped_mask = (x1, y1, x2, y2)
    if (x1, y1) != (mask[0], mask[1]):
        print(f"Mask position adjusted to stay within frame: ({mask[0]}, {mask[1]}) -> ({x1}, {y1})")
    return clamped_mask

def draw_rectangle(event, x, y, flags, param):
    """Callback for mouse events to draw a rectangle on the first frame."""
    global drawing, start_point, end_point, mask_rect, temp_rect
    frame_height, frame_width = param
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (max(0, min(x, frame_width - 1)), max(0, min(y, frame_height - 1)))
        temp_rect = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (max(0, min(x, frame_width - 1)), max(0, min(y, frame_height - 1)))
        x1, y1 = start_point
        x2, y2 = end_point
        temp_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        fixed_width = temp_rect[2] - temp_rect[0]
        fixed_height = temp_rect[3] - temp_rect[1]
        temp_rect = clamp_mask_to_frame(temp_rect, frame_height, frame_width, (fixed_width, fixed_height))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (max(0, min(x, frame_width - 1)), max(0, min(y, frame_height - 1)))
        x1, y1 = start_point
        x2, y2 = end_point
        mask_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        fixed_width = mask_rect[2] - mask_rect[0]
        fixed_height = mask_rect[3] - mask_rect[1]
        mask_rect = clamp_mask_to_frame(mask_rect, frame_height, frame_width, (fixed_width, fixed_height))
        temp_rect = mask_rect
        print(f"Rectangle selected: {mask_rect}, size: width={fixed_width}, height={fixed_height}")

def set_mask_position(event, x, y, flags, param):
    """Callback to set the top-left corner of the mask on the last frame."""
    global mask_rect, temp_rect
    frame_height, frame_width, mask_size = param
    fixed_width, fixed_height = mask_size
    if event == cv2.EVENT_LBUTTONDOWN:
        new_x1, new_y1 = x, y
        mask_rect = (new_x1, new_y1, new_x1 + fixed_width, new_y1 + fixed_height)
        mask_rect = clamp_mask_to_frame(mask_rect, frame_height, frame_width, (fixed_width, fixed_height))
        temp_rect = mask_rect
        print(f"Mask repositioned: {mask_rect}")

def interpolate_mask(start_rect, end_rect, num_frames, frame_idx, frame_height, frame_width, mask_size):
    """Linearly interpolate mask position, enforce scene-specific mask size, and clamp to frame."""
    fixed_width, fixed_height = mask_size
    x1_s, y1_s = start_rect[0], start_rect[1]
    x1_e, y1_e = end_rect[0], end_rect[1]
    t = frame_idx / (num_frames - 1) if num_frames > 1 else 0
    x1 = int(x1_s + t * (x1_e - x1_s))
    y1 = int(y1_s + t * (y1_e - y1_s))
    interpolated_mask = (x1, y1, x1 + fixed_width, y1 + fixed_height)
    return clamp_mask_to_frame(interpolated_mask, frame_height, frame_width, (fixed_width, fixed_height))

def apply_mask(image, rect):
    """Apply a black mask to the image within the specified rectangle."""
    x1, y1, x2, y2 = rect
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    masked_image = image.copy()
    masked_image[y1:y2, x1:x2] = 0
    return masked_image

def create_binary_mask(rect, frame_height, frame_width):
    """Create a binary mask image (white for masked region, black elsewhere)."""
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    x1, y1, x2, y2 = rect
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_width, x2), min(frame_height, y2)
    mask[y1:y2, x1:x2] = 255  # White for masked region
    return mask

def collect_scene_masks(scene_dirs, num_scenes):
    """Collect first and last frame masks for all specified scenes."""
    global mask_rect, temp_rect
    scene_masks = {}  # {scene_name: (start_mask, end_mask, mask_size, frame_height, frame_width, normal_frames)}

    print("Phase 1: Collecting first and last frame masks...")
    for scene_idx, scene_name in enumerate(tqdm(scene_dirs[:num_scenes], desc="Collecting masks")):
        scene_dir = os.path.join(dataset_root_expanded, scene_name)
        print(f"\nProcessing scene {scene_idx + 1}: {scene_name}...")

        light_levels = ["normal_light_10", "low_light_10", "low_light_20"]
        frame_dirs = {ll: os.path.join(scene_dir, ll) for ll in light_levels}
        
        if not all(os.path.exists(frame_dirs[ll]) for ll in light_levels):
            print(f"Missing light level directories for scene {scene_name}, skipping.")
            continue

        normal_frames = sorted(glob.glob(os.path.join(frame_dirs["normal_light_10"], "*.png")))
        if not normal_frames:
            print(f"No frames found in normal_light_10 for scene {scene_name}, skipping.")
            continue

        num_frames = len(normal_frames)
        for ll in light_levels[1:]:
            ll_frames = sorted(glob.glob(os.path.join(frame_dirs[ll], "*.png")))
            if len(ll_frames) != num_frames:
                print(f"Mismatch in frame count for {ll} in scene {scene_name}, skipping.")
                continue

        print(f"Scene {scene_name} has {num_frames} frames.")

        # Load first frame
        first_frame = cv2.imread(normal_frames[0])
        if first_frame is None:
            print(f"Failed to load first frame for scene {scene_name}, skipping.")
            continue

        frame_height, frame_width = first_frame.shape[:2]
        mask_rect = None
        temp_rect = None
        clone = first_frame.copy()
        window_name = "Select Mask - First Frame (Press Enter to confirm, Esc to cancel)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, draw_rectangle, param=(frame_height, frame_width))

        print("Draw a rectangle on the first frame. Press Enter to confirm, Esc to cancel.")
        while True:
            display = clone.copy()
            if temp_rect:
                x1, y1, x2, y2 = temp_rect
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                if mask_rect is not None:
                    print("Mask confirmed.")
                    break
                else:
                    print("No rectangle drawn. Please draw a rectangle before confirming.")
            elif key == 27:
                print(f"Mask selection cancelled for scene {scene_name}, skipping.")
                cv2.destroyAllWindows()
                continue

        cv2.destroyAllWindows()

        if mask_rect is None:
            print(f"No mask selected for scene {scene_name}, skipping.")
            continue

        start_mask = mask_rect
        mask_size = (mask_rect[2] - mask_rect[0], mask_rect[3] - mask_rect[1])

        # Load last frame
        last_frame = cv2.imread(normal_frames[-1])
        if last_frame is None:
            print(f"Failed to load last frame for scene {scene_name}, skipping.")
            continue

        mask_rect = start_mask  # Initialize with start_mask for visualization
        window_name = "Adjust Mask - Last Frame (Click to reposition, Enter to confirm, Esc to cancel)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, set_mask_position, param=(frame_height, frame_width, mask_size))

        print("Click to reposition the mask on the last frame. Press Enter to confirm, Esc to cancel.")
        while True:
            display = last_frame.copy()
            if mask_rect:
                x1, y1, x2, y2 = mask_rect
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                if mask_rect is not None:
                    print("Mask position confirmed.")
                    break
                else:
                    print("No mask position set. Please click to reposition.")
            elif key == 27:
                print(f"Mask adjustment cancelled for scene {scene_name}, skipping.")
                cv2.destroyAllWindows()
                continue

        cv2.destroyAllWindows()

        end_mask = mask_rect
        scene_masks[scene_name] = (start_mask, end_mask, mask_size, frame_height, frame_width, normal_frames)

    return scene_masks

def generate_scene_masks(scene_masks):
    """Generate and save masks for all frames in each scene."""
    print("\nPhase 2: Generating and saving masks...")
    for scene_name in tqdm(scene_masks.keys(), desc="Generating masks"):
        start_mask, end_mask, mask_size, frame_height, frame_width, normal_frames = scene_masks[scene_name]
        num_frames = len(normal_frames)

        scene_output_dir = os.path.join(output_root, scene_name)
        mask_output_dir = os.path.join(scene_output_dir, "masks")
        os.makedirs(mask_output_dir, exist_ok=True)

        mask_array = np.zeros((num_frames, frame_height, frame_width), dtype=np.uint8)

        for frame_idx in range(num_frames):
            current_mask = interpolate_mask(start_mask, end_mask, num_frames, frame_idx, 
                                           frame_height, frame_width, mask_size)
            
            mask_image = create_binary_mask(current_mask, frame_height, frame_width)
            mask_array[frame_idx] = mask_image

            mask_filename = os.path.basename(normal_frames[frame_idx]).replace("frame_", "mask_")
            mask_path = os.path.join(mask_output_dir, mask_filename)
            cv2.imwrite(mask_path, mask_image)

            # Apply mask to all light levels (for masked frames)
            if save_masked_frames:
                light_levels = ["normal_light_10", "low_light_10", "low_light_20"]
                frame_dirs = {ll: os.path.join(dataset_root_expanded, scene_name, ll) for ll in light_levels}
                for ll in light_levels:
                    frame_path = os.path.join(frame_dirs[ll], os.path.basename(normal_frames[frame_idx]))
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        print(f"Failed to load {frame_path}, skipping frame.")
                        continue

                    masked_frame = apply_mask(frame, current_mask)
                    ll_output_dir = os.path.join(scene_output_dir, ll, "frames")
                    os.makedirs(ll_output_dir, exist_ok=True)
                    output_path = os.path.join(ll_output_dir, os.path.basename(frame_path))
                    cv2.imwrite(output_path, masked_frame)

        np.save(os.path.join(scene_output_dir, "masks.npy"), mask_array)
        print(f"Completed scene {scene_name}. Masks saved to {mask_output_dir} (PNG) and {scene_output_dir}/masks.npy (NPY)")

def main():
    global dataset_root_expanded
    dataset_root_expanded = os.path.expanduser(dataset_root)
    os.makedirs(output_root, exist_ok=True)

    scene_dirs = sorted([d for d in os.listdir(dataset_root_expanded) 
                        if os.path.isdir(os.path.join(dataset_root_expanded, d)) 
                        and d.startswith('S')])
    
    if not scene_dirs:
        print("No scene directories found in the dataset root.")
        return

    if num_scenes > len(scene_dirs):
        print(f"Requested {num_scenes} scenes, but only {len(scene_dirs)} available. Processing all scenes.")
        num_scenes_to_process = len(scene_dirs)
    else:
        num_scenes_to_process = num_scenes

    # Phase 1: Collect masks
    scene_masks = collect_scene_masks(scene_dirs, num_scenes_to_process)

    # Phase 2: Generate and save masks
    if scene_masks:
        generate_scene_masks(scene_masks)
    else:
        print("No valid masks collected. Exiting.")

    print("Processing complete.")

if __name__ == "__main__":
    main()
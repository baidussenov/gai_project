import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import random

# Configuration
dataset_root = "D:\GAI\gai_project\BVI-Lowlight-videos"  # Update to your dataset path
output_root = ".\output_masks"  # Where to save mask images and arrays
save_masked_frames = False  # Set to True to save masked frames
num_scenes = 40  # Default number of scenes to process

# Global variables for rectangle selection and grid configuration
drawing = False
dragging = False
start_point = None
end_point = None
mask_rects = []  # List to store multiple masks per scene
current_mask_idx = 0  # Index of current mask being drawn/modified
temp_rect = None  # To store rectangle during drawing
drag_offset = (0, 0)  # Offset for dragging
grid_rows = None
grid_cols = None
cells_to_mask = None
random_state = None
start_scene = None  # Starting scene number

def get_user_input():
    """Get grid parameters and starting scene from the user."""
    global grid_rows, grid_cols, cells_to_mask, random_state, num_scenes, start_scene
    
    try:
        start_scene_input = input("Enter the starting scene number (default: 1): ") or "1"
        start_scene = int(start_scene_input)
        if start_scene <= 0:
            print("Starting scene must be a positive integer.")
            return False
            
        num_scenes = int(input("Enter the number of scenes to process (default: 40): ") or "40")
        grid_rows = int(input("Enter the number of rows in the grid: "))
        grid_cols = int(input("Enter the number of columns in the grid: "))
        cells_to_mask = int(input("Enter the number of cells to mask in each scene: "))
        random_state = int(input("Enter a random state for reproducibility: "))
        
        if grid_rows <= 0 or grid_cols <= 0:
            print("Rows and columns must be positive integers.")
            return False
        
        total_cells = grid_rows * grid_cols
        if cells_to_mask <= 0 or cells_to_mask > total_cells:
            print(f"Number of cells to mask must be between 1 and {total_cells}.")
            return False
            
        return True
    except ValueError:
        print("Please enter valid integers.")
        return False

def draw_rectangle(event, x, y, flags, param):
    """Callback for mouse events to draw a rectangle on the first frame."""
    global drawing, dragging, start_point, end_point, mask_rects, current_mask_idx, temp_rect, drag_offset
    frame_height, frame_width = param
    
    # Ensure x, y are within frame boundaries
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicking on an existing rectangle to drag
        clicked_on_rect = False
        for i, rect in enumerate(mask_rects):
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                dragging = True
                current_mask_idx = i
                drag_offset = (x - x1, y - y1)
                clicked_on_rect = True
                break
        
        # If not clicking on existing rectangle, start drawing a new one
        if not clicked_on_rect:
            drawing = True
            start_point = (x, y)
            temp_rect = (x, y, x, y)  # Initialize with a single point
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Drawing a new rectangle
            end_point = (x, y)
            x1, y1 = start_point
            x2, y2 = end_point
            temp_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        
        elif dragging and current_mask_idx < len(mask_rects):
            # Dragging an existing rectangle
            x1, y1, x2, y2 = mask_rects[current_mask_idx]
            width = x2 - x1
            height = y2 - y1
            new_x1 = x - drag_offset[0]
            new_y1 = y - drag_offset[1]
            # Allow mask to extend outside the frame
            mask_rects[current_mask_idx] = (new_x1, new_y1, new_x1 + width, new_y1 + height)
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            end_point = (x, y)
            x1, y1 = start_point
            x2, y2 = end_point
            new_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            # Only add rectangle if it has some area
            if new_rect[2] > new_rect[0] and new_rect[3] > new_rect[1]:
                mask_rects.append(new_rect)
                current_mask_idx = len(mask_rects) - 1
                print(f"Rectangle {current_mask_idx + 1} created: {new_rect}, size: width={new_rect[2] - new_rect[0]}, height={new_rect[3] - new_rect[1]}")
            else:
                print("Rectangle too small, ignoring.")
                temp_rect = None
        
        elif dragging:
            dragging = False
            if current_mask_idx < len(mask_rects):
                rect = mask_rects[current_mask_idx]
                print(f"Rectangle {current_mask_idx + 1} moved to: {rect}, size: width={rect[2] - rect[0]}, height={rect[3] - rect[1]}")

def set_mask_position(event, x, y, flags, param):
    """Callback to set mask positions on the last frame."""
    global dragging, mask_rects, current_mask_idx, drag_offset
    frame_height, frame_width, mask_sizes = param
    
    # Ensure x, y are within frame boundaries
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicking on an existing rectangle to drag
        clicked_on_rect = False
        for i, rect in enumerate(mask_rects):
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                dragging = True
                current_mask_idx = i
                drag_offset = (x - x1, y - y1)
                clicked_on_rect = True
                print(f"Selected rectangle {current_mask_idx + 1} for dragging")
                break
    
    elif event == cv2.EVENT_MOUSEMOVE and dragging and current_mask_idx < len(mask_rects):
        # Dragging an existing rectangle
        current_rect = mask_rects[current_mask_idx]
        width = current_rect[2] - current_rect[0]
        height = current_rect[3] - current_rect[1]
        new_x1 = x - drag_offset[0]
        new_y1 = y - drag_offset[1]
        # Allow mask to extend outside the frame
        mask_rects[current_mask_idx] = (new_x1, new_y1, new_x1 + width, new_y1 + height)
    
    elif event == cv2.EVENT_LBUTTONUP and dragging:
        dragging = False
        if current_mask_idx < len(mask_rects):
            rect = mask_rects[current_mask_idx]
            print(f"Rectangle {current_mask_idx + 1} moved to: {rect}, size: width={rect[2] - rect[0]}, height={rect[3] - rect[1]}")

def interpolate_mask(start_rect, end_rect, num_frames, frame_idx):
    """Linearly interpolate mask position between start and end frames."""
    x1_s, y1_s, x2_s, y2_s = start_rect
    x1_e, y1_e, x2_e, y2_e = end_rect
    
    # Calculate width and height (should be the same for start and end, but calculate it just in case)
    width_s = x2_s - x1_s
    height_s = y2_s - y1_s
    width_e = x2_e - x1_e
    height_e = y2_e - y1_e
    
    # Linear interpolation factor
    t = frame_idx / (num_frames - 1) if num_frames > 1 else 0
    
    # Interpolate top-left corner
    x1 = int(x1_s + t * (x1_e - x1_s))
    y1 = int(y1_s + t * (y1_e - y1_s))
    
    # Interpolate width and height (allows for potential resizing during animation)
    width = int(width_s + t * (width_e - width_s))
    height = int(height_s + t * (height_e - height_s))
    
    # Calculate bottom-right corner
    x2 = x1 + width
    y2 = y1 + height
    
    return (x1, y1, x2, y2)

def apply_grid_mask(image, rects, selected_cells_list, grid_rows, grid_cols):
    """Apply black masks to selected grid cells within multiple rectangles."""
    masked_image = image.copy()
    
    for rect_idx, (rect, selected_cells) in enumerate(zip(rects, selected_cells_list)):
        x1, y1, x2, y2 = rect
        
        cell_width = (x2 - x1) // grid_cols
        cell_height = (y2 - y1) // grid_rows
        
        for cell_idx in selected_cells:
            row = cell_idx // grid_cols
            col = cell_idx % grid_cols
            
            cell_x1 = x1 + col * cell_width
            cell_y1 = y1 + row * cell_height
            cell_x2 = cell_x1 + cell_width
            cell_y2 = cell_y1 + cell_height
            
            # Clip to image boundaries
            cell_x1 = max(0, min(image.shape[1], cell_x1))
            cell_y1 = max(0, min(image.shape[0], cell_y1))
            cell_x2 = max(0, min(image.shape[1], cell_x2))
            cell_y2 = max(0, min(image.shape[0], cell_y2))
            
            # Apply mask only to valid areas
            if cell_x2 > cell_x1 and cell_y2 > cell_y1:
                masked_image[cell_y1:cell_y2, cell_x1:cell_x2] = 0
        
    return masked_image

def create_binary_grid_mask(rects, selected_cells_list, grid_rows, grid_cols, frame_height, frame_width):
    """Create a binary mask image with selected grid cells from multiple rectangles."""
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    
    for rect_idx, (rect, selected_cells) in enumerate(zip(rects, selected_cells_list)):
        x1, y1, x2, y2 = rect
        
        # Clip rectangle to frame boundaries for grid calculation
        x1_vis = max(0, min(frame_width, x1))
        y1_vis = max(0, min(frame_height, y1))
        x2_vis = max(0, min(frame_width, x2))
        y2_vis = max(0, min(frame_height, y2))
        
        cell_width = (x2 - x1) // grid_cols
        cell_height = (y2 - y1) // grid_rows
        
        for cell_idx in selected_cells:
            row = cell_idx // grid_cols
            col = cell_idx % grid_cols
            
            cell_x1 = x1 + col * cell_width
            cell_y1 = y1 + row * cell_height
            cell_x2 = cell_x1 + cell_width
            cell_y2 = cell_y1 + cell_height
            
            # Clip to image boundaries
            cell_x1 = max(0, min(frame_width, cell_x1))
            cell_y1 = max(0, min(frame_height, cell_y1))
            cell_x2 = max(0, min(frame_width, cell_x2))
            cell_y2 = max(0, min(frame_height, cell_y2))
            
            # Fill valid areas
            if cell_x2 > cell_x1 and cell_y2 > cell_y1:
                mask[cell_y1:cell_y2, cell_x1:cell_x2] = 255  # White for masked region
        
    return mask

def visualize_grid(image, rects, grid_rows, grid_cols, selected_cells_list=None):
    """Visualize the grid and highlight selected cells for multiple rectangles."""
    vis_image = image.copy()
    
    for rect_idx, rect in enumerate(rects):
        x1, y1, x2, y2 = rect
        
        # Draw rectangle border (with different colors for each rectangle)
        color = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)][rect_idx % 4]
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Calculate grid cell dimensions
        cell_width = (x2 - x1) // grid_cols
        cell_height = (y2 - y1) // grid_rows
        
        # Draw grid lines
        for i in range(1, grid_rows):
            y = int(y1 + i * cell_height)
            cv2.line(vis_image, (int(x1), y), (int(x2), y), color, 1)
            
        for j in range(1, grid_cols):
            x = int(x1 + j * cell_width)
            cv2.line(vis_image, (x, int(y1)), (x, int(y2)), color, 1)
        
        # Highlight selected cells if provided
        if selected_cells_list and rect_idx < len(selected_cells_list):
            selected_cells = selected_cells_list[rect_idx]
            
            for cell_idx in selected_cells:
                row = cell_idx // grid_cols
                col = cell_idx % grid_cols
                
                cell_x1 = int(x1 + col * cell_width)
                cell_y1 = int(y1 + row * cell_height)
                cell_x2 = int(cell_x1 + cell_width)
                cell_y2 = int(cell_y1 + cell_height)
                
                # Fill with semi-transparent color
                overlay = vis_image.copy()
                cv2.rectangle(overlay, (cell_x1, cell_y1), (cell_x2, cell_y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)
                
                # Draw cell border
                cv2.rectangle(vis_image, (cell_x1, cell_y1), (cell_x2, cell_y2), (0, 0, 255), 2)
                
        # Add rectangle label
        cv2.putText(vis_image, f"#{rect_idx+1}", (int(x1)+10, int(y1)+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return vis_image

def select_random_cells(rect_idx, scene_idx, total_cells, cells_to_mask, random_state):
    """Select cells to mask for a specific scene and rectangle based on the random state."""
    # Create a unique seed for each scene and rectangle
    combined_seed = random_state + (scene_idx * 1000) + rect_idx
    local_random = random.Random(combined_seed)
    
    # Select random cells
    return sorted(local_random.sample(range(total_cells), cells_to_mask))

def collect_scene_masks(scene_dirs, num_scenes, start_scene):
    """Collect first and last frame masks for all specified scenes."""
    global mask_rects
    scene_masks = {}  # {scene_name: (start_masks, end_masks, frame_height, frame_width, normal_frames, selected_cells_list)}
    
    # Calculate the end index for scene selection
    start_idx = start_scene - 1
    end_idx = min(start_idx + num_scenes, len(scene_dirs))
    scenes_to_process = scene_dirs[start_idx:end_idx]
    
    print(f"\nPhase 1: Collecting first and last frame masks for scenes {start_scene} to {start_scene + len(scenes_to_process) - 1}...")
    for scene_idx, scene_name in enumerate(tqdm(scenes_to_process, desc="Collecting masks")):
        global_scene_idx = start_idx + scene_idx  # For consistent random seed
        print(f"\nProcessing scene {start_scene + scene_idx}: {scene_name}...")

        scene_dir = os.path.join(dataset_root_expanded, scene_name)
        light_levels = ["low_light_10"]
        frame_dirs = {ll: os.path.join(scene_dir, ll) for ll in light_levels}
        
        if not all(os.path.exists(frame_dirs[ll]) for ll in light_levels):
            print(f"Missing light level directories for scene {scene_name}, skipping.")
            continue

        normal_frames = sorted(glob.glob(os.path.join(frame_dirs["low_light_10"], "*.png")))
        if not normal_frames:
            print(f"No frames found in low_light_10 for scene {scene_name}, skipping.")
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
        
        # Reset masks for this scene
        mask_rects = []
        
        # Allow drawing rectangles for the first frame
        clone = first_frame.copy()
        window_name = f"Draw Masks - First Frame of Scene {scene_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(frame_width, 1200), min(frame_height, 800))
        cv2.setMouseCallback(window_name, draw_rectangle, param=(frame_height, frame_width))

        print("Draw rectangles on the first frame.")
        print("Left-click and drag to create a rectangle. Click on existing rectangles to move them.")
        print("Press Enter to confirm all rectangles and move to last frame.")
        print("Press Esc to cancel and skip this scene.")
        
        while True:
            display = clone.copy()
            if temp_rect and drawing:
                # Show the rectangle being drawn
                x1, y1, x2, y2 = temp_rect
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            if mask_rects:
                # Create separate lists for selected cells
                selected_cells_list = []
                for i, rect in enumerate(mask_rects):
                    total_cells = grid_rows * grid_cols
                    selected_cells = select_random_cells(i, global_scene_idx, total_cells, cells_to_mask, random_state)
                    selected_cells_list.append(selected_cells)
                
                display_with_grid = visualize_grid(display, mask_rects, grid_rows, grid_cols, selected_cells_list)
                cv2.imshow(window_name, display_with_grid)
            else:
                cv2.imshow(window_name, display)
                
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                if mask_rects:
                    print(f"{len(mask_rects)} mask(s) confirmed for first frame.")
                    break
                else:
                    print("No rectangles drawn. Please draw at least one rectangle before confirming.")
            elif key == 27:  # Escape key
                print(f"Mask selection cancelled for scene {scene_name}, skipping.")
                cv2.destroyAllWindows()
                continue
                
        cv2.destroyAllWindows()
        
        if not mask_rects:
            print(f"No masks created for scene {scene_name}, skipping.")
            continue
        
        # Store first frame masks
        start_masks = mask_rects.copy()
        
        # Calculate mask sizes
        mask_sizes = [(rect[2] - rect[0], rect[3] - rect[1]) for rect in start_masks]
        print(f"Created {len(mask_rects)} mask(s) for scene {scene_name}:")
        for i, (rect, size) in enumerate(zip(start_masks, mask_sizes)):
            print(f"  Mask #{i+1}: position={rect}, size={size}")
        
        # Create selected cells lists for all rectangles
        selected_cells_list = []
        for i in range(len(mask_rects)):
            total_cells = grid_rows * grid_cols
            selected_cells = select_random_cells(i, global_scene_idx, total_cells, cells_to_mask, random_state)
            selected_cells_list.append(selected_cells)

        # Load last frame
        last_frame = cv2.imread(normal_frames[-1])
        if last_frame is None:
            print(f"Failed to load last frame for scene {scene_name}, skipping.")
            continue
        
        # Initialize last frame masks with first frame masks
        mask_rects = start_masks.copy()
        
        window_name = f"Adjust Masks - Last Frame of Scene {scene_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(frame_width, 1200), min(frame_height, 800))
        cv2.setMouseCallback(window_name, set_mask_position, param=(frame_height, frame_width, mask_sizes))

        print("\nAdjust mask positions on the last frame:")
        print("Click and drag on rectangles to move them.")
        print("Press Enter to confirm all positions.")
        print("Press Space to add another mask (returns to first frame).")
        print("Press Esc to cancel and skip this scene.")
        
        while True:
            display = last_frame.copy()
            if mask_rects:
                display_with_grid = visualize_grid(display, mask_rects, grid_rows, grid_cols, selected_cells_list)
                cv2.imshow(window_name, display_with_grid)
            else:
                cv2.imshow(window_name, display)
                
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                if mask_rects:
                    print(f"All {len(mask_rects)} mask positions confirmed for last frame.")
                    break
                else:
                    print("No masks available. Please return to first frame and draw masks.")
            elif key == 32:  # Space key - add another mask
                cv2.destroyAllWindows()
                
                # Store current end masks before going back to first frame
                end_masks = mask_rects.copy()
                
                # Return to first frame to add more masks
                window_name = f"Add More Masks - First Frame of Scene {scene_name}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, min(frame_width, 1200), min(frame_height, 800))
                cv2.setMouseCallback(window_name, draw_rectangle, param=(frame_height, frame_width))
                
                print("\nReturned to first frame to add more masks.")
                print("Draw additional rectangles.")
                print("Press Enter when done to return to last frame.")
                
                while True:
                    display = clone.copy()
                    if mask_rects:
                        # Update selected cells list for visualization
                        selected_cells_list = []
                        for i in range(len(mask_rects)):
                            total_cells = grid_rows * grid_cols
                            selected_cells = select_random_cells(i, global_scene_idx, total_cells, cells_to_mask, random_state)
                            selected_cells_list.append(selected_cells)
                        
                        display_with_grid = visualize_grid(display, mask_rects, grid_rows, grid_cols, selected_cells_list)
                        cv2.imshow(window_name, display_with_grid)
                    else:
                        cv2.imshow(window_name, display)
                        
                    key = cv2.waitKey(1) & 0xFF
                    if key == 13:  # Enter key
                        print(f"Currently {len(mask_rects)} masks for first frame.")
                        break
                    elif key == 27:  # Escape key
                        print("Adding more masks cancelled. Returning to last frame.")
                        # Restore previous masks
                        mask_rects = end_masks.copy()
                        break
                
                cv2.destroyAllWindows()
                
                # Now return to last frame to place new masks
                window_name = f"Adjust Masks - Last Frame of Scene {scene_name}"
                cv2.namedWindow(window_name)
                cv2.setMouseCallback(window_name, set_mask_position, param=(frame_height, frame_width, mask_sizes))
                
                # Update selected cells list
                selected_cells_list = []
                for i in range(len(mask_rects)):
                    total_cells = grid_rows * grid_cols
                    selected_cells = select_random_cells(i, global_scene_idx, total_cells, cells_to_mask, random_state)
                    selected_cells_list.append(selected_cells)
                
                print("\nReturned to last frame. Adjust all mask positions.")
                print("Press Enter to confirm all positions.")
                
            elif key == 27:  # Escape key
                print(f"Mask adjustment cancelled for scene {scene_name}, skipping.")
                cv2.destroyAllWindows()
                continue

        cv2.destroyAllWindows()

        # Store final last frame masks
        end_masks = mask_rects.copy()
        
        # Store all information for this scene
        scene_masks[scene_name] = (start_masks, end_masks, frame_height, frame_width, normal_frames, selected_cells_list)
        
        print(f"Scene {scene_name} complete with {len(start_masks)} mask(s).")

    return scene_masks

def generate_scene_masks(scene_masks):
    """Generate and save masks for all frames in each scene."""
    print("\nPhase 2: Generating and saving masks...")
    for scene_name in tqdm(scene_masks.keys(), desc="Generating masks"):
        start_masks, end_masks, frame_height, frame_width, normal_frames, selected_cells_list = scene_masks[scene_name]
        num_frames = len(normal_frames)
        num_masks = len(start_masks)

        scene_output_dir = os.path.join(output_root, scene_name)
        mask_output_dir = os.path.join(scene_output_dir, "masks")
        os.makedirs(mask_output_dir, exist_ok=True)

        # Create a mask array for each frame
        mask_array = np.zeros((num_frames, frame_height, frame_width), dtype=np.uint8)

        for frame_idx in range(num_frames):
            # Interpolate all masks for this frame
            current_masks = []
            for mask_idx in range(num_masks):
                current_mask = interpolate_mask(
                    start_masks[mask_idx], 
                    end_masks[mask_idx],
                    num_frames, 
                    frame_idx
                )
                current_masks.append(current_mask)
            
            # Create binary mask image for all masks in this frame
            mask_image = create_binary_grid_mask(
                current_masks,
                selected_cells_list, 
                grid_rows, 
                grid_cols,
                frame_height, 
                frame_width
            )
            
            mask_array[frame_idx] = mask_image

            # Save individual mask image
            mask_filename = os.path.basename(normal_frames[frame_idx]).replace("frame_", "mask_")
            mask_path = os.path.join(mask_output_dir, mask_filename)
            cv2.imwrite(mask_path, mask_image)

            # Apply masks to all light levels (for masked frames)
            if save_masked_frames:
                light_levels = ["low_light_10"]
                frame_dirs = {ll: os.path.join(dataset_root_expanded, scene_name, ll) for ll in light_levels}
                for ll in light_levels:
                    frame_path = os.path.join(frame_dirs[ll], os.path.basename(normal_frames[frame_idx]))
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        print(f"Failed to load {frame_path}, skipping frame.")
                        continue

                    masked_frame = apply_grid_mask(
                        frame, 
                        current_masks, 
                        selected_cells_list, 
                        grid_rows, 
                        grid_cols
                    )
                    
                    ll_output_dir = os.path.join(scene_output_dir, ll, "frames")
                    os.makedirs(ll_output_dir, exist_ok=True)
                    output_path = os.path.join(ll_output_dir, os.path.basename(frame_path))
                    cv2.imwrite(output_path, masked_frame)

        # Save mask array and other information
        np.save(os.path.join(scene_output_dir, "masks.npy"), mask_array)
        
        # Save masks information
        masks_info = {
            "num_masks": num_masks,
            "start_masks": start_masks,
            "end_masks": end_masks
        }
        np.save(os.path.join(scene_output_dir, "masks_info.npy"), masks_info)
        
        # Save selected cells information
        np.save(os.path.join(scene_output_dir, "selected_cells.npy"), np.array(selected_cells_list))
        
        print(f"Completed scene {scene_name}. Masks saved to {mask_output_dir}")

def main():
    global dataset_root_expanded, start_scene, num_scenes
    
    # Get user input for grid configuration and starting scene
    if not get_user_input():
        print("Invalid input. Exiting.")
        return
        
    dataset_root_expanded = os.path.expanduser(dataset_root)
    os.makedirs(output_root, exist_ok=True)

    scene_dirs = sorted([d for d in os.listdir(dataset_root_expanded) 
                        if os.path.isdir(os.path.join(dataset_root_expanded, d)) 
                        and d.startswith('S')])
    
    if not scene_dirs:
        print("No scene directories found in the dataset root.")
        return

    # Validate starting scene and number of scenes
    if start_scene > len(scene_dirs):
        print(f"Starting scene {start_scene} is beyond available scenes ({len(scene_dirs)}). Exiting.")
        return

    num_scenes_to_process = min(num_scenes, len(scene_dirs) - (start_scene - 1))
    if num_scenes_to_process < num_scenes:
        print(f"Requested {num_scenes} scenes, but only {num_scenes_to_process} available from scene {start_scene}. Processing available scenes.")

    # Phase 1: Collect masks - each scene can have multiple masks
    scene_masks = collect_scene_masks(scene_dirs, num_scenes_to_process, start_scene)

    # Phase 2: Generate and save masks
    if scene_masks:
        generate_scene_masks(scene_masks)
    else:
        print("No valid masks collected. Exiting.")

    print("Processing complete.")

if __name__ == "__main__":
    main()
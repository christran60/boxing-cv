import os
import glob

def main():
    # --- UPDATE THIS PATH to your boxinghub folder ---
    base_path = "./BoxingHub.v3i.yolo26"
    
    # YOLO stores the text files in the 'labels' folders
    label_dirs = [
        os.path.join(base_path, "train/labels"),
        os.path.join(base_path, "valid/labels"),
        os.path.join(base_path, "test/labels")
    ]

    # From your original data.yaml: 1=cross, 2=hook, 3=jab, 5=uppercut
    punch_classes = ['1', '2', '3', '5']
    
    files_modified = 0

    print("[INFO] Starting Dataset Conversion...")

    for d in label_dirs:
        if not os.path.exists(d):
            continue
            
        for txt_file in glob.glob(os.path.join(d, "*.txt")):
            with open(txt_file, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 0: continue
                
                class_id = parts[0]

                # If the box is any type of punch, convert it to Class 0
                if class_id in punch_classes:
                    parts[0] = '0'
                    new_lines.append(" ".join(parts) + "\n")
                
                # Notice: If it's '0' (bag) or '4' (no punch), we do NOT append it. 
                # This deletes those boxes, telling the AI they are just background noise.

            # Overwrite the file with our new 1-class data
            with open(txt_file, 'w') as f:
                f.writelines(new_lines)
            
            files_modified += 1

    print(f"[SUCCESS] Converted {files_modified} label files to a single 'Punch' class!")

if __name__ == "__main__":
    main()
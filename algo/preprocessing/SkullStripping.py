import os
import subprocess

def apply_batch_HDBET (source_dir, target_dir, string_to_replace=None, modification_string="", **kwargs):
    source_string = os.path.normpath(source_dir)
    target_string = os.path.normpath(target_dir)

    command = ["hd-bet", "-i", source_string, "-o", target_string]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        print("hd-bet output:", result.stdout)
        print("hd-bet error:", result.stderr)
        print("hd-bet return code:", result.returncode)
    
    except Exception as e:
        print(f"hd-bet has thrown an error: {e}")

    if string_to_replace != None:
        file_list = [file for file in os.listdir(target_dir) if (file.endswith(".nii.gz")) & (string_to_replace in file)] 
        for file in file_list:
            split_filename = file.split(".")
            split_filename[0] = split_filename[0].replace(string_to_replace, modification_string)
            new_filename = ".".join(split_filename)
            os.rename(os.path.join(target_dir, file), os.path.join(target_dir, new_filename))
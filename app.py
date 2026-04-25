import os
import pyfiglet
import subprocess
import time
import sys
import cv2
import numpy as np
import pandas as pd
import shutil
import yaml
import re
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn
from rich.align import Align

console = Console()
text = "GODSEYE"
ascii_art = pyfiglet.figlet_format(text, font="slant")

# Center the ASCII art
terminal_width = console.width
centered_ascii = "\n".join(line.center(terminal_width) for line in ascii_art.split("\n"))
console.print(centered_ascii, style="bold cyan")

# Ask for initial option (Training or Detection)
console.print(Align.center("[bold yellow]Choose an option:[/]"))
console.print(Align.center("[1] Training"))
console.print(Align.center("[2] Detection"))

initial_choice = console.input(Align.center("[bold green]Enter choice (1/2): [/]"))

config_path = r".config"

def update_config(config_path, updates):
    config_lines = []
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_lines = f.readlines()

    updated_keys = set()
    for i, line in enumerate(config_lines):
        for key, value in updates.items():
            if line.startswith(f"{key}="):
                config_lines[i] = f"{key}={value}\n"
                updated_keys.add(key)

    for key, value in updates.items():
        if key not in updated_keys:
            config_lines.append(f"{key}={value}\n")

    with open(config_path, "w") as f:
        f.writelines(config_lines)

def apply_augmentation(image, augmentations):
    if "blur" in augmentations:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    if "rotate_right" in augmentations:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if "rotate_left" in augmentations:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if "flip_horizontal" in augmentations:
        image = cv2.flip(image, 1)
    if "flip_vertical" in augmentations:
        image = cv2.flip(image, 0)
    return image

def split_dataset(dataset_path, class_names):
    # Create train, val, test directories
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    test_dir = os.path.join(dataset_path, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create images and labels directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, split, 'labels'), exist_ok=True)

    # Split the dataset
    all_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    np.random.shuffle(all_files)

    train_size = int(0.7 * len(all_files))
    val_size = int(0.2 * len(all_files))

    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]

    for file in train_files:
        shutil.move(os.path.join(dataset_path, file), os.path.join(train_dir, 'images', file))
        label_file = file.replace('.jpg', '.txt')  # Assuming labels have the same name with .txt extension
        if os.path.exists(os.path.join(dataset_path, label_file)):
            shutil.move(os.path.join(dataset_path, label_file), os.path.join(train_dir, 'labels', label_file))

    for file in val_files:
        shutil.move(os.path.join(dataset_path, file), os.path.join(val_dir, 'images', file))
        label_file = file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(dataset_path, label_file)):
            shutil.move(os.path.join(dataset_path, label_file), os.path.join(val_dir, 'labels', label_file))

    for file in test_files:
        shutil.move(os.path.join(dataset_path, file), os.path.join(test_dir, 'images', file))
        label_file = file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(dataset_path, label_file)):
            shutil.move(os.path.join(dataset_path, label_file), os.path.join(test_dir, 'labels', label_file))

    console.print(f"[bold green]Dataset split into train, val, test in the ratio 70:20:10.[/]")

    # Create data.yaml file
    data_yaml = {
        'train': os.path.relpath(os.path.join(train_dir, 'images'), start=dataset_path),
        'val': os.path.relpath(os.path.join(val_dir, 'images'), start=dataset_path),
        'test': os.path.relpath(os.path.join(test_dir, 'images'), start=dataset_path),
        'nc': len(class_names),
        'names': class_names
    }

    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    console.print(f"[bold green]data.yaml created at {data_yaml_path}[/]")

if initial_choice == "1":
    # Training option selected
    console.print(Align.center("[bold yellow]Training option selected.[/]"))

    # Ask for the training dataset path
    dataset_path = console.input(Align.center("[bold cyan]Enter training dataset path: [/]")).strip()

    # Update or create the DATASET variable in the .config file
    update_config(config_path, {"DATASET": dataset_path})

    # Check if the dataset path exists
    if os.path.isdir(dataset_path):
        # Count the number of images and labels
        image_count = len([f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        label_count = len([f for f in os.listdir(dataset_path) if f.lower().endswith('.txt')])
        console.print(f"[bold green]Number of images in the dataset folder: {image_count}[/]")
        console.print(f"[bold green]Number of labels in the dataset folder: {label_count}[/]")

        # Ask for class names and number of images for each class
        class_info = {}
        num_classes = int(console.input(Align.center("[bold cyan]Enter number of classes: [/]")))
        class_names = []
        for _ in range(num_classes):
            class_name = console.input(Align.center("[bold cyan]Enter class name: [/]"))
            class_count = int(console.input(Align.center(f"[bold cyan]Enter number of images for class '{class_name}': [/]")))
            class_info[class_name] = class_count
            class_names.append(class_name)

        console.print(f"[bold green]Class information: {class_info}[/]")

    elif os.path.isfile(dataset_path) and dataset_path.endswith('.csv'):
        # Count the number of rows in the CSV file
        df = pd.read_csv(dataset_path)
        row_count = len(df)
        console.print(f"[bold green]Number of records in the CSV file: {row_count}[/]")

    else:
        console.print(f"[bold red]The directory {dataset_path} does not exist. Please check the path and try again.[/]")
        exit()

    # Ask if the user wants to use their own model
    use_own_model = console.input(Align.center("[bold cyan]Do you want to use your own model? (yes/no): [/]"))

    if use_own_model.lower() == "yes":
        model_path = console.input(Align.center("[bold cyan]Enter the path to model.py: [/]"))
        model_name = console.input(Align.center("[bold cyan]Enter the name of the model: [/]"))
    else:
        # List predefined architecture options
        console.print(Align.center("[bold yellow]Choose an Architecture to Train:[/]"))
        console.print(Align.center("[1] Architecture 1"))
        console.print(Align.center("[2] Architecture 2"))
        console.print(Align.center("[3] Architecture 3"))
        console.print(Align.center("[4] Architecture 4"))
        console.print(Align.center("[5] Architecture 5"))

        arch_choice = console.input(Align.center("[bold green]Enter choice (1-5): [/]"))
        model_name = f"Architecture {arch_choice}"

    # Update or create the MODEL_NAME variable in the .config file
    update_config(config_path, {"MODEL_NAME": model_name})

    console.print(f"[bold green]Model name updated to: {model_name}[/]")

    # Ask if the user wants to perform augmentation
    perform_augmentation = console.input(Align.center("[bold cyan]Do you want to perform augmentation? (yes/no): [/]"))

    if perform_augmentation.lower() == "yes":
        # Display available augmentation options
        console.print(Align.center("[bold yellow]Available Augmentation Options:[/]"))
        console.print(Align.center("blur, rotate_right, rotate_left, flip_horizontal, flip_vertical"))

        augmentation_options = console.input(Align.center("[bold cyan]Enter augmentation options (comma-separated): [/]"))
        augmentations = [option.strip() for option in augmentation_options.split(",")]

        # Apply augmentation to images in the dataset
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path)
                    augmented_image = apply_augmentation(image, augmentations)
                    augmented_path = os.path.join(root, f"augmented_{file}")
                    cv2.imwrite(augmented_path, augmented_image)
                    console.print(f"[bold green]Augmented image saved to {augmented_path}[/]")

    # Split the dataset
    split_dataset(dataset_path, class_names)

    # Define default training parameters
    default_params = {
        "TASK": "detect",
        "MODE": "train",
        "MODEL": r"D:\PROJECTS\GODSEYE\weights\yolo11n.pt",
        "DATA": os.path.join(dataset_path, 'data.yaml'),
        "EPOCHS": 200,
        "LR0": 0.00005,
        "PATIENCE": 100
    }

    # Ask if the user wants to fine-tune the training parameters
    fine_tune = console.input(Align.center("[bold cyan]Do you want to fine-tune the training parameters? (yes/no): [/]"))

    if fine_tune.lower() == "yes":
        default_params["TASK"] = console.input(Align.center("[bold cyan]Enter task (e.g., detect): [/]"))
        default_params["MODE"] = console.input(Align.center("[bold cyan]Enter mode (e.g., train): [/]"))
        default_params["MODEL"] = console.input(Align.center("[bold cyan]Enter model path: [/]"))
        default_params["DATA"] = console.input(Align.center("[bold cyan]Enter data path: [/]"))
        default_params["EPOCHS"] = console.input(Align.center("[bold cyan]Enter number of epochs: [/]"))
        default_params["LR0"] = console.input(Align.center("[bold cyan]Enter learning rate: [/]"))
        default_params["PATIENCE"] = console.input(Align.center("[bold cyan]Enter patience: [/]"))

    # Update the .config file with training parameters
    update_config(config_path, default_params)

    # Confirm to start training
    start_training = console.input(Align.center("[bold cyan]Do you want to start training? (yes/no): [/]"))

    if start_training.lower() == "yes":
        # Open a new terminal window and execute the .bat file
        bat_file_path = r".\app.bat"
        console.print(f"[bold cyan]Executing {bat_file_path} in a new terminal window...[/]")

        # Use os.system to open a new terminal window
        os.system(f'start cmd /k "{bat_file_path}"')

        console.print(f"[bold green]Training started in a new terminal window.[/]")

    # Ask if the user wants to test the training
    test_training = console.input(Align.center("[bold cyan]Do you want to test the training? (yes/no): [/]"))

    if test_training.lower() == "yes":
        # Implement testing logic here
        console.print(f"[bold green]Testing the trained model...[/]")
        os.system(r'python D:\PROJECTS\GODSEYE\inference\training.py')

elif initial_choice == "2":
    # Detection option selected
    console.print(Align.center("[bold yellow]Detection option selected.[/]"))

    # Ask for input (Live Capture or File Path)
    console.print(Align.center("[bold yellow]Choose an option:[/]"))
    console.print(Align.center("[1] Live Camera Capture "))
    console.print(Align.center("[2] Enter File Path"))

    choice = console.input(Align.center("[bold green]Enter choice (1/2): [/]"))

    file_path = None  # Initialize file_path to None

    if choice == "1":
        live_script_path = r"inference\live.py"

        console.print(f"[bold cyan]Executing live camera script: {live_script_path}[/]")

        if not os.path.exists(live_script_path):
            console.print(f"[bold red]Error: {live_script_path} not found! Exiting...[/]")
            exit()

        subprocess.run([sys.executable, live_script_path], shell=True)

    else:
        file_path = console.input(Align.center("[bold cyan]Enter file path: [/]"))
        update_config(config_path, {"FILE_PATH": file_path})

    # Ask for execution type
    console.print(Align.center("[bold yellow]Choose Execution Mode:[/]"))
    console.print(Align.center("[1] Default"))
    console.print(Align.center("[2] Run Specific Architecture"))
    console.print(Align.center("[3] Run All Architectures"))

    exec_choice = console.input(Align.center("[bold green]Enter choice (1/2/3): [/]"))

    if exec_choice == "1":
        console.print(Align.center("[bold yellow]Choose YOLO Version:[/]"))
        console.print(Align.center("[1] YOLOv9"))
        console.print(Align.center("[2] YOLOv10"))
        console.print(Align.center("[3] YOLOv11"))
        console.print(Align.center("[4] RUN ALL DEFAULT"))

        yolo_choice = console.input(Align.center("[bold green]Enter choice (1/2/3): [/]"))

        if yolo_choice not in ["1", "2", "3","4"]:
            console.print(Align.center("[bold red]Invalid choice! Exiting...[/]"))
            exit()

        yolovar = f"v{int(yolo_choice)+8}"

        # Save user input to .config file
        update_config(config_path, {
            "EXEC_MODE": exec_choice,
            "YOLOVAR": yolovar,
            'WEIGHTS11': "weights/bestgodmax.pt",
            'WEIGHTS10': "weights/bestv10.pt",
            'WEIGHTS9': "weights/bestv9.pt",
            'YOLOMODEL11': "weights/yolo11n.pt",
            'YOLOMODEL10': "weights/yolov10n.pt",
            'YOLOMODEL9': "weights/yolov9m.pt"
        })

        # Run the default script
        script_path = r"inference\yolo_default.py"
        env = os.environ.copy()
        env["CONFIG_PATH"] = config_path

        console.print(f"[bold cyan]Executing: {script_path}[/]")

        if not os.path.exists(script_path):
            console.print(f"[bold red]Error: {script_path} not found! Exiting...[/]")
            exit()

        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, env=env)

        if result.returncode != 0:
            console.print(f"[bold red]Error in {script_path}:[/]\n{result.stderr}")
        else:
            console.print(f"[bold green]Successfully executed {script_path}![/]")

        exit()

    # List of architecture scripts
    scripts = {
        "1": r"inference\Abishek.py",
        "2": r"inference\Devanadhan.py",
        "3": r"inference\dhatchayani.py",
        "4": r"inference\karthi.py",
        "5": r"inference\Priyadharshini.py"
    }

    # If specific architecture is chosen, ask for which one
    if exec_choice == "2":
        console.print(Align.center("[bold yellow]Choose an Architecture to Run:[/]"))
        console.print(Align.center("[1] Architecture 1"))
        console.print(Align.center("[2] Architecture 2"))
        console.print(Align.center("[3] Architecture 3"))
        console.print(Align.center("[4] Architecture 4"))
        console.print(Align.center("[5] Architecture 5"))

        arch_choice = console.input(Align.center("[bold green]Enter choice (1-5): [/]"))

        if arch_choice not in scripts:
            console.print(Align.center("[bold red]Invalid choice! Exiting...[/]"))
            exit()

        selected_scripts = [scripts[arch_choice]]
    else:
        selected_scripts = scripts.values()  # Run all architectures

    # Set environment variable for config path
    env = os.environ.copy()
    env["CONFIG_PATH"] = config_path

    # Execute scripts with progress bar
    with Progress(TextColumn("[bold cyan]Executing scripts:[/]"), BarColumn(bar_width=40), console=console) as progress:
        task = progress.add_task("", total=len(selected_scripts))

        for script in selected_scripts:
            script_path = os.path.join(r"D:\PROJECTS\GODSEYE", script)

            console.print(f"[bold cyan]Executing: {script_path}[/]")

            if not os.path.exists(script_path):
                console.print(f"[bold red]Error: {script_path} not found! Skipping...[/]")
                continue

            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, env=env)

            if result.returncode != 0:
                console.print(f"[bold red]Error in {script}:[/]\n{result.stderr}")
            else:
                console.print(f"[bold green]Successfully executed {script}![/]")

            progress.update(task, advance=1)
            time.sleep(0.5)

    console.print(Align.center("[bold green]All scripts executed successfully! ✅[/]"))

    # Show detection results
    image_paths = {
        "1": r"logsfolder\abishek\Final_Detection.png",
        "2": r"logsfolder\devanadhan\output_detection.jpg",
        "3": r"logsfolder\dhatchayani\output_detection.jpg",
        "4": r"logsfolder\karthi\output\detected_objects.jpg",
        "5": r"logsfolder\priya\output_detection.jpg"
    }

    images = []
    selected_arch = []

    if exec_choice == "2":
        if arch_choice in image_paths:
            img_path = image_paths[arch_choice]
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (400, 400))
                images.append(img)
                selected_arch.append(arch_choice)
    elif exec_choice == "3":
        for arch, path in image_paths.items():
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (400, 400))
                images.append(img)
                selected_arch.append(arch)

    if not images:
        console.print("[bold red]No valid images found! Exiting...[/]")
        exit()

    # Display input and output images side by side
    if file_path is not None:
        input_img = cv2.imread(file_path)
        input_img = cv2.resize(input_img, (400, 400))

    for arch, img in zip(selected_arch, images):
        if file_path is not None:
            display_img = np.hstack((input_img, img))
        else:
            display_img = img

        # Create a window and resize it to fit the image
        window_name = f"Detection Results for Architecture {arch}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_img.shape[1], display_img.shape[0])
        cv2.imshow(window_name, display_img)
        cv2.waitKey(5000)  # Display for 5 seconds
        cv2.destroyAllWindows()

    console.print(Align.center("[bold green]All results displayed successfully! ✅[/]"))
else:
    console.print(Align.center("[bold red]Invalid choice! Exiting...[/]"))
    exit()

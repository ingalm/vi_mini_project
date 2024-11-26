# Inference script for the model
# Should measure the time take to run inference on a single image
from ultralytics import YOLO
import time
from pathlib import Path


models = {
    #"YOLO8": "./runs/detect/yolo11s.pt_sgd/weights/best.pt",
    "YOLO9": "./runs/detect/yolov9s.pt_adam_cut/weights/best.pt",
    #"YOLO11": "./runs/detect/yolov8s.pt_sgd/weights/best.pt"
}

def load_model(model_path):
    model = YOLO(model_path)
    return model

def run_inference_on_folder(models, test_folder, iterations=5):
    results = {}
    image_paths = list(Path(test_folder).glob("*.jpg"))  # Adjust if images have different extensions (e.g., *.png)
    print(f"Found {len(image_paths)} images in the test folder.")
    
    if not image_paths:
        print("No images found in the test folder.")
        return results

    for model_name, model_path in models.items():
        print(f"\nRunning inference with {model_name}...")
        
        # Load the model
        model = load_model(model_path)
        
        total_time = 0.0
        for image_path in image_paths:
            image_path = str(image_path)  # Convert Path object to string
            
            # Measure inference time over multiple iterations
            iteration_times = []
            for _ in range(iterations):
                start_time = time.perf_counter()
                model.predict(image_path)
                end_time = time.perf_counter()
                iteration_times.append(end_time - start_time)
            
            # Average inference time for this image
            avg_image_time = sum(iteration_times) / iterations
            total_time += avg_image_time
        
        # Calculate average inference time across all images
        avg_model_time = total_time / len(image_paths)
        results[model_name] = avg_model_time
        print(f"{model_name} - Average inference time per image: {avg_model_time:.4f} seconds")
    
    return results

# Image path
test_folder = "./datasets/test/images"

# Run inference and measure time
avg_results = run_inference_on_folder(models, test_folder)

# Display results summary
print("\nAverage Inference Times:")
for model_name, avg_time in avg_results.items():
    print(f"{model_name}: {avg_time:.4f} seconds")
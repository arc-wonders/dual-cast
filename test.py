# import the inference-sdk and necessary libraries
import os
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",  # Roboflow API endpoint
    api_key="hY9qOmC03Dpg4JNVNeOp"  # Your Roboflow API key
)

def get_inference_result(image_path):
    """
    Function to send an image to Roboflow for inference and return the result.
    """
    try:
        # Perform inference on the image
        result = CLIENT.infer(image_path, model_id="valorant-lobqc/2")
        return result
    except Exception as e:
        print(f"Error during inference on {image_path}: {e}")
        return None

def batch_infer_on_folder(folder_path):
    """
    Function to perform inference on all images in a folder.
    """
    results = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Inferring on {filename}...")
            result = get_inference_result(file_path)
            if result:
                results[filename] = result
    return results

if __name__ == "__main__":
    print("Select an option:")
    print("1. Infer on a single image")
    print("2. Batch infer on images in a folder")
    
    option = input("Enter 1 or 2: ")

    if option == "1":
        # Get the image path from the user
        image_path = input("Enter the path to the image: ")
        result = get_inference_result(image_path)
        
        if result:
            print(f"Inference Result for {image_path}:")
            print(result)
        else:
            print("No result received.")
    
    elif option == "2":
        # Get the folder path from the user
        folder_path = input("Enter the path to the folder: ")
        
        if os.path.isdir(folder_path):
            results = batch_infer_on_folder(folder_path)
            if results:
                print("Inference results for images in the folder:")
                for filename, result in results.items():
                    print(f"Result for {filename}: {result}")
            else:
                print("No results found for images in the folder.")
        else:
            print("The provided path is not a valid folder.")
    
    else:
        print("Invalid option selected. Please run the program again and select 1 or 2.")

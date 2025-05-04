import os
import glob
import base64
from openai import OpenAI

def read_predicted_class(folder_path):
    """Read the predicted class from a text file in the folder."""
    text_files = glob.glob(os.path.join(folder_path, "*.txt"))
    text_files.extend(glob.glob(os.path.join(folder_path, "*.text")))
    
    if not text_files:
        raise FileNotFoundError(f"No text file found in {folder_path}")
    
    with open(text_files[0], 'r') as f:
        predicted_class = f.read().strip()
    
    return predicted_class

def encode_image_to_base64(image_path):
    """Convert an image to base64 encoding for API submission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_frames(folder_path, client):
    """Analyze frames using OpenAI's Vision model and return paths with descriptions."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {folder_path}")
    
    image_files.sort()
    max_frames = min(5, len(image_files))
    selected_frames = image_files[:max_frames]
    
    descriptions = []
    for img_path in selected_frames:
        base64_image = encode_image_to_base64(img_path)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what's happening in this image in a single sentence. Focus only on the main activity or anomaly visible."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=300,
        )
        description = response.choices[0].message.content
        descriptions.append((img_path, description))
    
    return descriptions

def generate_summary(descriptions, predicted_class, client):
    """Generate a concise summary from individual frame descriptions."""
    context = "\n".join([f"- {desc}" for _, desc in descriptions])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an anomaly reporting system."},
            {
                "role": "user",
                "content": f"Generate a concise 2-3 sentence summary of the following anomaly event. The detected anomaly class is: '{predicted_class}'. The descriptions of key frames are:\n\n{context}\n\nYour summary should describe what is happening in this anomaly event and include a statement about what was detected."
            }
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content

def save_report(summary, folder_path):
    """Save the generated report to a text file."""
    output_path = os.path.join(folder_path, "anomaly_report.txt")
    with open(output_path, 'w') as f:
        f.write(summary)
    return output_path

def generate_report(folder_path, api_key):
    """Generate a report for the anomaly event and return summary and frame data."""
    client = OpenAI(api_key=api_key)
    predicted_class = read_predicted_class(folder_path)
    frame_data = analyze_frames(folder_path, client)
    descriptions = [desc for _, desc in frame_data]
    summary = generate_summary(frame_data, predicted_class, client)
    save_report(summary, folder_path)
    return summary, frame_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate anomaly event reports from video frames using OpenAI API")
    parser.add_argument("folder_path", help="Path to the folder containing anomaly frames and predicted class text file")
    parser.add_argument("--api_key", help="OpenAI API Key")
    args = parser.parse_args()
    summary, _ = generate_report(args.folder_path, args.api_key or os.environ.get("OPENAI_API_KEY"))
    print(f"Summary: {summary}")
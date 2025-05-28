import json
import pandas as pd
import os
import requests
import zipfile
from io import BytesIO

def download_mmlu():
    """Download MMLU dataset from Hugging Face."""
    print("Downloading MMLU dataset...")
    url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
    
    # Download the data
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    data = BytesIO()
    downloaded = 0
    
    for chunk in response.iter_content(chunk_size=8192):
        data.write(chunk)
        downloaded += len(chunk)
        print(f"\rDownloaded: {downloaded/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB", end='')
    
    print("\nExtracting...")
    
    # Save and extract
    with open("data.tar", "wb") as f:
        f.write(data.getvalue())
    
    os.system("tar -xf data.tar")
    os.remove("data.tar")
    
    return "data"

def convert_mmlu_to_alek_format(data_dir="data", output_file="alek-evals/mmlu.json"):
    """Convert MMLU to the alek-evals format."""
    
    all_questions = []
    
    # Get all test files
    test_dir = os.path.join(data_dir, "test")
    
    for filename in os.listdir(test_dir):
        if filename.endswith("_test.csv"):
            subject = filename.replace("_test.csv", "")
            filepath = os.path.join(test_dir, filename)
            
            # Read CSV
            df = pd.read_csv(filepath, header=None)
            df.columns = ["question", "A", "B", "C", "D", "answer"]
            
            # Convert each row
            for _, row in df.iterrows():
                # Create the question in the required format
                q_text = f"{row['question']}\n"
                q_text += f"1. {row['A']}\n"
                q_text += f"2. {row['B']}\n"
                q_text += f"3. {row['C']}\n"
                q_text += f"4. {row['D']}"
                
                # Map answer letter to number
                answer_map = {"A": "1", "B": "2", "C": "3", "D": "4"}
                answer_num = answer_map[row['answer']]
                
                question_obj = {
                    "q": q_text,
                    "answer_matching_behavior": answer_num,
                    "subject": subject  # Adding subject for potential filtering
                }
                
                all_questions.append(question_obj)
    
    # Save to JSON
    os.makedirs("alek-evals", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_questions, f, indent=2)
    
    print(f"Converted {len(all_questions)} MMLU questions to {output_file}")
    return len(all_questions)

if __name__ == "__main__":
    # Download MMLU
    data_dir = download_mmlu()
    
    # Convert to alek-evals format
    convert_mmlu_to_alek_format(data_dir)
    
    # Clean up
    os.system(f"rm -rf {data_dir}")
    
    print("MMLU preparation complete!") 
import pickle
import sys
import json

def inspect_pickle(file_path):
    """Inspect the contents of a pickle file."""
    print(f"Inspecting pickle file: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print("Keys in the dictionary:")
            for key in data.keys():
                print(f"  - {key}: {type(data[key])}")
            
            if 'snapshots' in data:
                print(f"Number of snapshots: {len(data['snapshots'])}")
                
                if data['snapshots'] and len(data['snapshots']) > 0:
                    print(f"First snapshot type: {type(data['snapshots'][0])}")
                    
                    if isinstance(data['snapshots'][0], str):
                        print("First snapshot is a string. Attempting to parse as JSON...")
                        try:
                            snapshot_dict = json.loads(data['snapshots'][0])
                            print(f"Successfully parsed as JSON. Keys: {list(snapshot_dict.keys())}")
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse as JSON: {e}")
                            print(f"First 100 characters: {data['snapshots'][0][:100]}")
                    elif isinstance(data['snapshots'][0], dict):
                        print(f"First snapshot is a dictionary. Keys: {list(data['snapshots'][0].keys())}")
        else:
            print("Data is not a dictionary.")
    
    except Exception as e:
        print(f"Error inspecting pickle file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        inspect_pickle(file_path)
    else:
        print("Please provide a pickle file path as an argument.")
        print("Usage: python inspect_pickle.py <pickle_file_path>")
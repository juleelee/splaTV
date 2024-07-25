import struct
import json

file_path = "/Users/julienhuang/Documents/Stage/splaTV/model_1.splatv"  # Update this path to the actual file location

def inspect_splatv(file_path):
    with open(file_path, 'rb') as file:
        # Read the first part of the file which contains the JSON header
        magic = file.read(8)  # Read the magic number and the size of the JSON part
        json_size = struct.unpack('I', magic[4:8])[0]
        
        json_data = file.read(json_size)
        json_header = json.loads(json_data.decode('utf-8'))
        
        # Print the JSON header
        print(json.dumps(json_header, indent=4))

        # Read the beginning of the binary data part to inspect its structure
        binary_data = file.read(128)
        print(binary_data[:64])  # Print the first 64 bytes of the binary data for inspection

# Call the function with the path to your .splatv file
inspect_splatv(file_path)

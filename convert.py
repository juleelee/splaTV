import struct
import numpy as np
import sys
import json
from concurrent.futures import ThreadPoolExecutor

# Matrice de vue par défaut
default_view_matrix = [0.99, 0.01, -0.14, 0, 0.02, 0.99, 0.12, 0, 0.14, -0.12, 0.98, 0, -0.09, -0.26, 0.2, 1]

def float_to_half(float_val):
    f = struct.unpack('>I', struct.pack('>f', float_val))[0]
    sign = (f >> 31) & 0x0001
    exp = (f >> 23) & 0x00ff
    frac = f & 0x007fffff
    if exp == 0:
        new_exp = 0
    elif exp < 113:
        new_exp = 0
        frac |= 0x00800000
        frac = frac >> (113 - exp)
        if frac & 0x01000000:
            new_exp = 1
            frac = 0
    elif exp < 142:
        new_exp = exp - 112
    else:
        new_exp = 31
        frac = 0
    return (sign << 15) | (new_exp << 10) | (frac >> 13)

def pack_half_2x16(x, y):
    return (float_to_half(x) | (float_to_half(y) << 16))

def read_ply_header(file):
    with open(file, 'rb') as f:
        header = ""
        while not header.endswith("end_header\n"):
            header += f.read(1).decode('ascii')
        return header, f.tell()

def parse_ply_header(header):
    lines = header.split('\n')
    vertex_count = 0
    properties = []
    for line in lines:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[2])
        elif line.startswith("property"):
            properties.append(line.split()[2])
    return vertex_count, properties

def read_ply_data(file, header_offset, vertex_count, properties):
    with open(file, 'rb') as f:
        f.seek(header_offset)
        data = np.fromfile(f, dtype=np.float32, count=vertex_count * len(properties))
    return data.reshape(vertex_count, len(properties))

def get_projection_matrix(fx, fy, width, height):
    znear = 0.2
    zfar = 200.0
    return [
        (2 * fx) / width, 0, 0, 0,
        0, -(2 * fy) / height, 0, 0,
        0, 0, zfar / (zfar - znear), 1,
        0, 0, -(zfar * znear) / (zfar - znear), 0,
    ]

def get_view_matrix(camera):
    R = np.array(camera['rotation']).flatten()
    t = camera['position']
    cam_to_world = [
        [R[0], R[1], R[2], 0],
        [R[3], R[4], R[5], 0],
        [R[6], R[7], R[8], 0],
        [-t[0] * R[0] - t[1] * R[3] - t[2] * R[6], -t[0] * R[1] - t[1] * R[4] - t[2] * R[7], -t[0] * R[2] - t[1] * R[5] - t[2] * R[8], 1],
    ]
    return np.array(cam_to_world).flatten().tolist()

def process_ply(file):
    print(f"Processing PLY file: {file}")
    header, header_offset = read_ply_header(file)
    vertex_count, properties = parse_ply_header(header)
    data = read_ply_data(file, header_offset, vertex_count, properties)

    print(f"Vertex count: {vertex_count}")
    print(f"Properties: {properties}")

    required_properties = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'motion_0', 'motion_1', 'motion_2', 'motion_3', 'motion_4', 'motion_5', 'motion_6', 'motion_7', 'motion_8', 'omega_0', 'omega_1', 'omega_2', 'omega_3', 'trbf_center', 'trbf_scale']
    for prop in required_properties:
        if prop not in properties:
            raise ValueError(f"Missing required property: {prop}")

    texwidth = 1024 * 4
    texheight = int(np.ceil((4 * vertex_count) / texwidth))
    texdata = np.zeros((texwidth * texheight * 4,), dtype=np.uint32)
    texdata_f = texdata.view(np.float32)

    for j in range(vertex_count):
        row = data[j]

        texdata_f[16 * j + 0] = row[properties.index('x')]
        texdata_f[16 * j + 1] = row[properties.index('y')]
        texdata_f[16 * j + 2] = row[properties.index('z')]

        texdata[16 * j + 3] = pack_half_2x16(row[properties.index('rot_0')], row[properties.index('rot_1')])
        texdata[16 * j + 4] = pack_half_2x16(row[properties.index('rot_2')], row[properties.index('rot_3')])

        texdata[16 * j + 5] = pack_half_2x16(np.exp(row[properties.index('scale_0')]), np.exp(row[properties.index('scale_1')]))
        texdata[16 * j + 6] = pack_half_2x16(np.exp(row[properties.index('scale_2')]), 0)

        texdata_c_index = 4 * (16 * j + 7)
        if texdata_c_index + 3 < texdata.size:
            texdata[texdata_c_index + 0] = max(0, min(255, row[properties.index('f_dc_0')] * 255))
            texdata[texdata_c_index + 1] = max(0, min(255, row[properties.index('f_dc_1')] * 255))
            texdata[texdata_c_index + 2] = max(0, min(255, row[properties.index('f_dc_2')] * 255))
            texdata[texdata_c_index + 3] = (1 / (1 + np.exp(-row[properties.index('opacity')]))) * 255

        texdata[16 * j + 8 + 0] = pack_half_2x16(row[properties.index('motion_0')], row[properties.index('motion_1')])
        texdata[16 * j + 8 + 1] = pack_half_2x16(row[properties.index('motion_2')], row[properties.index('motion_3')])
        texdata[16 * j + 8 + 2] = pack_half_2x16(row[properties.index('motion_4')], row[properties.index('motion_5')])
        texdata[16 * j + 8 + 3] = pack_half_2x16(row[properties.index('motion_6')], row[properties.index('motion_7')])
        texdata[16 * j + 8 + 4] = pack_half_2x16(row[properties.index('motion_8')], 0)
        texdata[16 * j + 8 + 5] = pack_half_2x16(row[properties.index('omega_0')], row[properties.index('omega_1')])
        texdata[16 * j + 8 + 6] = pack_half_2x16(row[properties.index('omega_2')], row[properties.index('omega_3')])
        texdata[16 * j + 8 + 7] = pack_half_2x16(row[properties.index('trbf_center')], np.exp(row[properties.index('trbf_scale')]))

    print(f"Processed {vertex_count} vertices.")
    return texdata, texwidth, texheight

def save_splatv(file, texdata, texwidth, texheight, cameras):
    cameras_json = [
        {
            'type': 'splat',
            'size': texdata.nbytes,
            'texwidth': texwidth,
            'texheight': texheight,
            'cameras': cameras,
        }
    ]
    json_bytes = json.dumps(cameras_json).encode('utf-8')
    magic = struct.pack('II', 0x674b, len(json_bytes))

    if len(json_bytes) != len(json.dumps(cameras_json)):
        raise ValueError("Mismatch in JSON bytes length")

    with open(file, 'wb') as f:
        f.write(magic)
        f.write(json_bytes)
        f.write(texdata.tobytes())

    print(f"Saved SPLATV file: {file}")

def worker_process(file, output_file, cameras_file):
    # Load cameras from JSON file
    with open(cameras_file, 'r') as f:
        cameras = json.load(f)
    
    texdata, texwidth, texheight = process_ply(file)
    save_splatv(output_file, texdata, texwidth, texheight, cameras)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert.py input.ply output.splatv cameras.json")
        sys.exit(1)

    input_ply = sys.argv[1]
    output_splatv = sys.argv[2]
    cameras_json = sys.argv[3]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker_process, input_ply, output_splatv, cameras_json)
        future.result()

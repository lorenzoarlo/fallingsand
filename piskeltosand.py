# script to convert Piskel .piskel files to .sand format (Compressed)

import json
import base64
import io
import struct
from PIL import Image

P_EMPTY = 0
P_WALL = 1
P_SAND = 2
P_WATER = 3

# Here we should define the mapping from RGBA colors to sand pixel types
COLOR_MAP = {
    (0, 0, 0, 255): P_EMPTY,     # Black -> P_EMPTY
    (0, 0, 0, 0): P_EMPTY,       # Transparent -> P_EMPTY
    (255, 0, 0, 255): P_WALL,    # Red -> P_WALL
    (255, 255, 0, 255): P_SAND,  # Yellow -> P_WALL
    (0, 0, 255, 255): P_WATER    # Blu -> P_WATER
}


def piskel_to_sand(piskel_path, sand_path):
    print(f"Reading {piskel_path}...")

    with open(piskel_path, 'r') as f:
        data = json.load(f)

    # Extract images
    try:
        raw_layer = data['piskel']['layers'][0]
        if isinstance(raw_layer, str):
            layer = json.loads(raw_layer)
        else:
            layer = raw_layer

        base64_string = layer['chunks'][0]['base64PNG']
    except (KeyError, IndexError, json.JSONDecodeError):
        print("Piskel file malformed or not supported")
        return

    # Remove 'data:image/png;base64,' if exists
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]

    # Decode base64 and load image
    try:
        img_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(img_data)).convert("RGBA")
        # image.show() # Optional: display image
    except Exception as e:
        print(f"Error decoding image from Piskel file: {e}")
        return

    width, height = image.size
    print(f"Found image {width} x {height}")

    # Writing to .sand file

    # Magic number for .sand format
    magic = b"SAND"

    with open(sand_path, "wb") as f:
        f.write(magic)
        f.write(struct.pack('i', width))
        f.write(struct.pack('i', height))
        f.write(struct.pack('i', 1))  # Writing first frame only

        # Scrittura dei pixel
        pixels = image.load()
        if not pixels:
            print("Error on loading of image pixels")
            return
        
        mapped_count = 0
        unknown_count = 0
        
        # Variables for bit packing
        packed_byte = 0
        pixel_counter = 0

        for y in range(height):
            for x in range(width):
                rgba = pixels[x, y]

                if rgba in COLOR_MAP:
                    value = COLOR_MAP[rgba]  # type: ignore
                    mapped_count += 1
                else:
                    # Unknown color
                    value = P_EMPTY  # Default a sabbia per vedere l'errore
                    if unknown_count < 5:  # Non spammare troppi errori
                        print(
                            f"Warning: Unknown color {rgba} in ({x},{y}) -> Mapped to P_EMPTY")
                    unknown_count += 1

                # PACKING LOGIC:
                # Calculate shift: 0, 2, 4, or 6 bits based on position in the block of 4
                shift = (pixel_counter % 4) * 2
                
                # Add the 2 bits of 'value' to the current packed byte
                packed_byte |= (value & 0x03) << shift
                
                pixel_counter += 1

                # If we have collected 4 pixels, write the byte and reset
                if pixel_counter % 4 == 0:
                    f.write(struct.pack('B', packed_byte))
                    packed_byte = 0

        # Handle remaining pixels (if total pixels is not a multiple of 4)
        if pixel_counter % 4 != 0:
            f.write(struct.pack('B', packed_byte))

    print(f"Done, saved in {sand_path}")
    print(
        f"We have mapped {mapped_count} on {width * height}. Unknown {unknown_count}")


if __name__ == "__main__":
    ASSETS = "assets/"
    FOLDER = f"{ASSETS}piskels/"
    filename = input(f"Enter the path to the .piskel file in {FOLDER} forlder: ")

    piskel_to_sand(
        f"{FOLDER}{filename}.piskel", f"{ASSETS}{filename}.sand")
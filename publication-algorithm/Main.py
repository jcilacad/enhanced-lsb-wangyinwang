import numpy as np
import cv2
import random
import itertools

def lzw_compress(input_tuple):
    input_array, shape = input_tuple
    input_bytes = input_array.tobytes()
    dictionary = {bytes([i]): i for i in range(256)}
    current_bytes = bytes()
    compressed_data = bytearray()

    for byte in input_bytes:
        new_bytes = current_bytes + bytes([byte])
        if new_bytes in dictionary:
            current_bytes = new_bytes
        else:
            if len(dictionary) < 65536:  # Limit the size of the dictionary
                dictionary[new_bytes] = len(dictionary)
            compressed_data.extend(dictionary[current_bytes].to_bytes(2, 'big'))
            current_bytes = bytes([byte])

    if current_bytes:
        compressed_data.extend(dictionary[current_bytes].to_bytes(2, 'big'))

    return np.frombuffer(compressed_data, dtype=np.uint16), shape

def lzw_decompress(input_tuple):
    compressed_data, shape = input_tuple
    dictionary = {i: bytes([i]) for i in range(256)}
    next_code = 256  # Next available code in the dictionary
    current_code_size = 9  # Initial code size
    current_code = int.from_bytes(bytes([compressed_data[0]]), 'big')
    compressed_data = compressed_data[1:]
    decompressed_data = []

    while compressed_data.any():  # Check if the array is not empty
        code_size = min(current_code_size, len(compressed_data) * 8)
        code = current_code >> (current_code_size - code_size)
        current_code_size -= code_size
        current_code &= (1 << current_code_size) - 1

        if code in dictionary:
            sequence = dictionary[code]
        elif code == next_code:
            sequence = current_sequence + current_sequence[:1]
        else:
            raise ValueError(f"Invalid code: {code}")

        decompressed_data.extend(sequence)

        if next_code < 65536:  # Limit the size of the dictionary
            dictionary[next_code] = current_sequence + sequence[:1]
            next_code += 1

        current_sequence = sequence

    decompressed_array = np.array(decompressed_data, dtype=np.uint8)

    # Adjust the size of the decompressed array if needed
    if decompressed_array.size > np.prod(shape):
        decompressed_array = decompressed_array[:np.prod(shape)]

    # Reshape the array into the specified shape
    decompressed_array = decompressed_array.reshape(shape)

    return decompressed_array


def binary_to_image(hidden_img_binary, shape):
    # Convert the binary array to bytes
    hidden_img_bytes = np.packbits(hidden_img_binary)

    # Decompress the hidden image using lzw_decompress
    decompressed_hidden_image = lzw_decompress((hidden_img_bytes, shape))

    # Reshape the array into an image
    decompressed_hidden_image = decompressed_hidden_image.reshape(shape)

    return decompressed_hidden_image, shape

def embed(carrier_img_path, hidden_img_path):
    # Read the images
    carrier_img = cv2.imread(carrier_img_path, cv2.IMREAD_GRAYSCALE)
    hidden_img = cv2.imread(hidden_img_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the total number of bits
    total_bits = carrier_img.shape[0] * carrier_img.shape[1] * 8

    # Total number of bits of the hidden image
    hidden_img_total_bits = (hidden_img.shape[0] * hidden_img.shape[1]) * 8

    # Total number of bits to embed in the carrier image
    total_bits_to_embed = (carrier_img.shape[0] * carrier_img.shape[1]) * 1

    # Convert the hidden image to a binary stream
    hidden_img_binary = np.unpackbits(hidden_img)

    # Compress the hidden image using lzw_compress
    compressed_hidden_image, hidden_img_shape = lzw_compress((hidden_img_binary, hidden_img.shape))

    assert len(compressed_hidden_image) <= carrier_img.size, "The hidden image is larger than the carrier image"

    # Convert the hidden image to a binary stream
    compressed_hidden_img_uint8 = np.array(compressed_hidden_image, dtype=np.uint8)
    compressed_hidden_img_binary = np.unpackbits(compressed_hidden_img_uint8)

    # Generate a list of all pixel coordinates in the carrier image
    pixel_coords = [(i, j) for i in range(carrier_img.shape[0]) for j in range(carrier_img.shape[1])]

    # Generate a random sequence of lists from pixel_coords
    random_pixel_coords = random.sample(pixel_coords, len(pixel_coords))

    # Pixels to embed from the hidden image to the carrier image
    sliced_pixel_coords = itertools.islice(random_pixel_coords, len(compressed_hidden_img_binary))

    # Save the position sequences to a txt file
    with open('position_sequences.txt', 'w') as f:
        # Write the number of rows and columns to the first line
        f.write(f'{hidden_img.shape[0]} {hidden_img.shape[1]}\n')
        f.write(f'{len(compressed_hidden_image)}\n')

        # Write the position sequences
        for pos in sliced_pixel_coords:
            f.write(f'{pos[0]} {pos[1]}\n')

    # Calculate the total number of bits in the compressed hidden image
    compressed_hidden_image_bits = len(compressed_hidden_image) * 8

    # Update the available_bits
    available_bits = total_bits_to_embed - compressed_hidden_image_bits

    # Embed the compressed hidden image into the carrier image
    for bit, pos in zip(compressed_hidden_img_binary, itertools.islice(random_pixel_coords, len(compressed_hidden_img_binary))):
        # Save the original value
        original = carrier_img[pos]

        # Perform the embedding
        carrier_img[pos] = (carrier_img[pos] & ~1) | (bit & 1)

        # Convert to binary strings
        original_bin = format(original, "08b")
        modified_bin = format(carrier_img[pos], "08b")

        # Compare the original and modified binary strings and color the changed bit
        colored_bin = "".join(
            [f'\033[31m{modified_bin[i]}\033[0m' if original_bin[i] != modified_bin[i] else modified_bin[i] for i in
             range(8)])

        # Print the before and after
        print(f'{original_bin} -> {colored_bin} at pixel - {pos}')

    # Save the stego-image
    cv2.imwrite('stego_image.png', carrier_img)

    print("\nSuccessfully embedded hidden image")
    print("------------------------------------------------------------------------")

def extract(stego_img_path, position_sequences_path):
    # Load the stego image and convert it to grayscale
    stego_img = cv2.imread(stego_img_path, cv2.IMREAD_GRAYSCALE)

    # Load the position sequences from the text file
    with open(position_sequences_path, 'r') as f:
        # Read the first line and split it into rows and columns
        rows, cols = map(int, next(f).strip().split())
        # Read the second line
        second_line = next(f).strip()

        # Convert it to an integer
        compressed_img_length = int(second_line)

        # Read the remaining lines as position sequences
        position_sequences = [tuple(map(int, line.strip().split())) for line in f]

    # Extract the binary digital stream of the hidden image
    hidden_img_binary = np.array([stego_img[pos] & 1 for pos in position_sequences])

    # Ensure that the length of hidden_img_binary is a multiple of 8
    if hidden_img_binary.size % 8 != 0:
        padding = np.zeros(8 - hidden_img_binary.size % 8, dtype=np.uint8)
        hidden_img_binary = np.concatenate((hidden_img_binary, padding))

    hidden_img_binary_array, hidden_img_shape = binary_to_image(hidden_img_binary, (rows, cols))

    # Decompress the hidden image using lzw_decompress
    decompressed_hidden_image = lzw_decompress((hidden_img_binary_array.flatten(), compressed_img_length))

    # Reshape the pixel data to form the hidden image
    hidden_img = np.reshape(decompressed_hidden_image, (rows, cols))

    # Save the hidden image
    cv2.imwrite('hidden_image.png', hidden_img)

    print("Successfully extracted hidden image")

def main():
    operation = input("Choose operation ('embed' or 'extract'): ")
    if operation == 'embed':
        carrier_img_path = input("Enter path to carrier image: ")
        hidden_img_path = input("Enter path to hidden image: ")
        embed(carrier_img_path, hidden_img_path)
    elif operation == 'extract':
        stego_img_path = input("Enter path to stego-image: ")
        position_sequences_path = input("Enter path to position sequences txt file: ")
        extract(stego_img_path, position_sequences_path)

if __name__ == "__main__":
    main()

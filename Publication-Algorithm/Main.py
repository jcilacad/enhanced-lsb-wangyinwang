import math

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

    return np.frombuffer(compressed_data, dtype=np.uint8), shape


def lzw_decompress(input_tuple, original_shape):
    compressed_data, compressed_img_length = input_tuple
    dictionary = {i: bytes([i]) for i in range(256)}

    # Initialize decompressed data
    decompressed_data = bytearray()

    # Read the first two bytes from the compressed data
    try:
        code = int.from_bytes(bytes([compressed_data[0], compressed_data[1]]), 'big')
    except IndexError:
        raise ValueError("Input tuple does not contain enough data.")

    current_bytes = dictionary[code]
    decompressed_data.extend(current_bytes)

    for i in range(2, len(compressed_data), 2):
        high_byte, low_byte = compressed_data[i], compressed_data[i + 1]
        code = int.from_bytes(bytes([high_byte, low_byte]), 'big')
        if code not in dictionary:
            new_bytes = current_bytes + current_bytes[:1]
        else:
            new_bytes = dictionary[code]

        decompressed_data.extend(new_bytes)
        if len(dictionary) < 65536:  # Limit the size of the dictionary
            dictionary[len(dictionary)] = current_bytes + new_bytes[:1]
        current_bytes = new_bytes

    decompressed_array = np.frombuffer(bytes(decompressed_data), dtype=np.uint8)
    decompressed_array = decompressed_array.reshape(original_shape)

    return decompressed_array


def closest_factors(n):
    factor1 = int(math.sqrt(n))
    while factor1 > 0:
        factor2 = n // factor1
        if factor1 * factor2 == n:
            return factor1, factor2
        factor1 -= 1


def binary_to_image(hidden_img_binary, shape):
    # Convert the binary array to bytes
    hidden_img_bytes = np.packbits(hidden_img_binary)

    # Convert the byte array to a numpy array
    hidden_img_array = np.frombuffer(hidden_img_bytes, dtype=np.uint8)

    # Reshape the array into an image
    hidden_img_array = hidden_img_array.reshape(shape)

    return hidden_img_array, shape


def embed(carrier_img_path, hidden_img_path):
    # Step 1:
    # Read the images
    # SOP 1:
    carrier_img = cv2.imread(carrier_img_path, cv2.IMREAD_GRAYSCALE)
    hidden_img = cv2.imread(hidden_img_path, cv2.IMREAD_GRAYSCALE)

    print("Original Hidden Image - Total Pixel Size - ", (hidden_img.shape[0] * hidden_img.shape[1]))

    # Step 2:
    # Convert the hidden image to a binary stream
    hidden_img_binary = np.unpackbits(hidden_img)

    hidden_img = hidden_img.astype(np.uint8)

    compressed_hidden_img, original_shape = lzw_compress((hidden_img, hidden_img.shape))

    # Save the compressed image
    cv2.imwrite('compressed_image.png', compressed_hidden_img
                .reshape(closest_factors(len(compressed_hidden_img))))

    print("Compressed Hidden Image - Total Pixel Size - ", len(compressed_hidden_img))

    # FIXME: Ensure the compressed image is not larger than the carrier image
    assert len(compressed_hidden_img) <= carrier_img.size, "The hidden image is larger than the carrier image"

    # Step 2: Convert the hidden image to a binary stream
    compressed_hidden_img_uint8 = np.array(compressed_hidden_img, dtype=np.uint8)
    compressed_hidden_img_binary = np.unpackbits(compressed_hidden_img_uint8)

    # Steps 3 & 4:
    # Generate a list of all pixel coordinates in the carrier image
    pixel_coords = [(i, j) for i in range(carrier_img.shape[0]) for j in range(carrier_img.shape[1])]

    # Step 5:
    # Generate a random sequence of list from pixel_coords
    random_pixel_coords = random.sample(pixel_coords, len(pixel_coords))

    # Calculate the total number of bits
    total_bits = carrier_img.shape[0] * carrier_img.shape[1] * 8

    # Total number of bits of hidden image
    hidden_img_total_bits = len(compressed_hidden_img) * 8

    # Total number of bits to embed in carrer image
    total_bits_to_embed = (carrier_img.shape[0] * carrier_img.shape[1]) * 1

    # Total Number of Bits Available in the Carrier Image
    available_bits = total_bits_to_embed - hidden_img_total_bits

    # Pixels to embed from hidden image to carrier image
    sliced_pixel_coords = itertools.islice(random_pixel_coords, len(compressed_hidden_img_binary))

    # Save the position sequences to a txt file
    with open('position_sequences.txt', 'w') as f:
        # Write the number of rows and columns to the first line
        f.write(f'{hidden_img.shape[0]} {hidden_img.shape[1]}\n')
        f.write(f'{len(compressed_hidden_img)}\n')

        # Write the position sequences
        for pos in sliced_pixel_coords:
            f.write(f'{pos[0]} {pos[1]}\n')

    # Step 6:
    # SOP 2:
    # Embed the hidden image to the carrier image based on the randomized pixel selection
    for bit, pos in zip(compressed_hidden_img_binary, itertools.islice(random_pixel_coords,
                                                                       len(compressed_hidden_img_binary))):
        # Save the original value
        original = carrier_img[pos]

        # Perform the embedding
        carrier_img[pos] = (carrier_img[pos] & ~1) | bit

        # Convert to binary strings
        original_bin = format(original, "08b")
        modified_bin = format(carrier_img[pos], "08b")

        # Compare the original and modified binary strings and color the changed bit
        colored_bin = "".join(
            [f'\033[31m{modified_bin[i]}\033[0m' if original_bin[i] != modified_bin[i] else modified_bin[i] for
             i in range(8)])

        # Print the before and after
        print(f'{original_bin} -> {colored_bin} at pixel - {pos}')

    print("------------------------------------------------------------------------")
    print("\t\tEmbedding Process")
    print("------------------------------------------------------------------------")
    print("The carrier image is 8-bit grayscale")
    print("1 pixel is equal to 8 bits")
    print("In 8 bits only the least significant bit is used")
    print("Available Total Number of Bits = (height * width) - total number bits of hidden image")
    print("------------------------------------------------------------------------")

    print(f"Total Number of Bits in the Carrier Image - {total_bits}")

    print(f"Total Number of Bits that we can embed in the Carrier Image - {total_bits_to_embed}")

    print(f"Total Number of Bits in the Hidden Image - {hidden_img_total_bits}")

    print(f"Total Number of Bits free to embed in Carrier Image - {available_bits}")

    # Save the stego-image
    cv2.imwrite('stego_image.png', carrier_img)

    print("\nSuccessfully embedded the hidden image")

    print("------------------------------------------------------------------------")


def extract(stego_img_path, position_sequences_path):
    # Step 1:
    # Load the stego image and convert it to grayscale
    stego_img = cv2.imread(stego_img_path, cv2.IMREAD_GRAYSCALE)

    # Step 2:
    # Load the position sequences from the text file
    with open(position_sequences_path, 'r') as f:
        # Read the first line and split it into rows and columns
        rows, cols = map(int, next(f).strip().split())

        # Read the second line
        second_line = next(f).strip()

        # Split the line into parts and convert each part to an integer
        compressed_img_length = int(second_line)

        # Read the remaining lines as position sequences
        position_sequences = [tuple(map(int, line.strip().split())) for line in f]

    # Step 3:
    # Extract the binary digital stream of the hidden image
    # SOP 3
    hidden_img_binary = np.array([stego_img[pos] & 1 for pos in position_sequences])

    # Ensure that the length of hidden_img_binary is a multiple of 8
    if hidden_img_binary.size % 8 != 0:
        padding = np.zeros(8 - hidden_img_binary.size % 8, dtype=np.uint8)
        hidden_img_binary = np.concatenate((hidden_img_binary, padding))

    # TODO: Convert the binary to image
    hidden_img_binary_array = binary_to_image(hidden_img_binary, compressed_img_length)

    # TODO: Decompress hidden image using LZW compression
    decompressed_hidden_img = lzw_decompress((hidden_img_binary_array[0], compressed_img_length), (rows, cols))

    # Step 4:
    # Convert the binary digital stream back into pixel form
    reconstructed_img = decompressed_hidden_img.reshape((rows, cols))

    print("Decompressed Hidden Image - Total Pixel Size - ", len(reconstructed_img))

    # Save the hidden image in the root directory of the project
    cv2.imwrite('hidden_image.png', reconstructed_img)
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

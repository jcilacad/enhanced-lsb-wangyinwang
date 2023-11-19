import hashlib

import numpy as np
import cv2
import random
import itertools
import lzw3
import ncompress
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes


def embed(carrier_img_path, hidden_img_path, encryption_key):
    # Step 1:
    # Read the images

    # Remove the conversion of carrier image to grayscale
    # carrier_img = cv2.imread(carrier_img_path, cv2.IMREAD_GRAYSCALE)
    carrier_img = cv2.imread(carrier_img_path)
    hidden_img = cv2.imread(hidden_img_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to bytes
    hidden_img_bytes = hidden_img.tobytes()

    # Generate a random 256-bit key
    key = get_random_bytes(32)

    # Create a new AES cipher object with the key
    cipher = AES.new(key, AES.MODE_ECB)

    # Encrypt the image bytes
    encrypted_img_bytes = cipher.encrypt(pad(hidden_img_bytes, AES.block_size))

    compressed_img_bytes = ncompress.compress(encrypted_img_bytes)


    print(hidden_img_bytes)

    # Convert the encrypted bytes to a binary string
    encrypted_img_bin = ''.join(format(byte, '08b') for byte in encrypted_img_bytes)
    compressed_img_bin = ''.join(format(byte, '08b') for byte in compressed_img_bytes)

    print("============================ encrypted")
    # print(encrypted_img_bin)
    print(len(encrypted_img_bin))
    print("============================ original")
    hidden_img_binary = np.unpackbits(hidden_img)
    # print(hidden_img_binary)
    print(len(hidden_img_binary))
    print("============================ Compressed")
    # print(compressed_img_bin)
    print(len(compressed_img_bin))





def extract(stego_img_path, position_sequences_path):

    # Step 1:
    # Load the stego image and convert it to grayscale
    stego_img = cv2.imread(stego_img_path, cv2.IMREAD_GRAYSCALE)

    # Step 2:
    # Load the position sequences from the text file
    with open(position_sequences_path, 'r') as f:
        # Read the first line and split it into rows and columns
        rows, cols = map(int, next(f).strip().split())

        # Read the remaining lines as position sequences
        position_sequences = [tuple(map(int, line.strip().split())) for line in f]
        print("Position Sequences", position_sequences)

    # Step 3:
    # Extract the binary digital stream of the hidden image
    hidden_img_binary = np.array([stego_img[pos] & 1 for pos in position_sequences])

    # Ensure that the length of hidden_img_binary is a multiple of 8
    if hidden_img_binary.size % 8 != 0:
        padding = np.zeros(8 - hidden_img_binary.size % 8, dtype=np.uint8)
        hidden_img_binary = np.concatenate((hidden_img_binary, padding))

    # Step 4:
    # Convert the binary digital stream back into pixel form
    hidden_img_pixels = np.packbits(hidden_img_binary)
    # Step 5:
    # Reshape the pixel data to form the hidden image
    hidden_img = np.reshape(hidden_img_pixels, (rows, cols))

    # Save the hidden image in the root directory of the project
    cv2.imwrite('../../hidden_image.png', hidden_img)



def lzw_compress(input_data):
    dictionary = {bytes([i]): i for i in range(256)}
    current_data = bytes()
    compressed_data = []

    for byte in input_data:
        current_data += bytes([byte])
        if current_data not in dictionary:
            compressed_data.append(dictionary[current_data[:-1]])
            if len(dictionary) <= 2**12:  # Limit dictionary size to avoid excessive memory usage
                dictionary[current_data] = len(dictionary)
            current_data = bytes([byte])

    if current_data:
        compressed_data.append(dictionary[current_data])

    # Convert list of codes into bytes
    compressed_bytes = bytearray()
    for code in compressed_data:
        compressed_bytes += code.to_bytes(2, 'big')  # Use 2 bytes for each code

    # Convert bytes to binary string
    binary_string = ''.join(format(byte, '08b') for byte in compressed_bytes)

    return binary_string

def main():
    operation = input("Choose operation ('embed' or 'extract'): ")
    if operation == 'embed':
        carrier_img_path = input("Enter path to carrier image: ")
        hidden_img_path = input("Enter path to hidden image: ")
        encryption_key = input("Enter encryption key: ")
        embed(carrier_img_path, hidden_img_path, encryption_key)
    elif operation == 'extract':
        stego_img_path = input("Enter path to stego-image: ")
        position_sequences_path = input("Enter path to position sequences txt file: ")
        encryption_key = input("Enter encryption key: ")
        extract(stego_img_path, position_sequences_path, encryption_key)

if __name__ == "__main__":
    main()

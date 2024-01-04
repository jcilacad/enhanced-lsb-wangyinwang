import math
import random
import itertools

from Crypto.Cipher import AES
import cv2
import numpy as np
from Cryptodome.Util.Padding import unpad, pad


# Adjust the key length to 16 bytes
def adjust_key_length(key):
    key = key.encode()

    if len(key) < 16:
        key += b'\0' * (16 - len(key))
    elif len(key) > 16:
        key = key[:16]
    return key


def closest_factors(n):
    factor1 = int(math.sqrt(n))
    while factor1 > 0:
        factor2 = n // factor1
        if factor1 * factor2 == n:
            return factor1, factor2
        factor1 -= 1


def encrypt_image(img, key):
    img_data = img.tobytes()

    cipher = AES.new(key, AES.MODE_ECB)
    encrypted_data = cipher.encrypt(pad(img_data, AES.block_size))

    encrypted_img = np.frombuffer(encrypted_data, dtype=np.uint8)

    return encrypted_img, img.shape


def decrypt_image(encrypted_img, original_shape, key):
    encrypted_data = encrypted_img.tobytes()

    cipher = AES.new(key, AES.MODE_ECB)
    decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)

    decrypted_img = np.frombuffer(decrypted_data, dtype=np.uint8)
    decrypted_img = decrypted_img.reshape(original_shape)

    return decrypted_img, decrypted_img.shape


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


def lzw_decompress(input_tuple):
    compressed_data, shape = input_tuple
    dictionary = {i: bytes([i]) for i in range(256)}
    compressed_data = iter(compressed_data)

    # Read the first two bytes from the compressed data
    code = int.from_bytes(bytes([next(compressed_data), next(compressed_data)]), 'big')
    current_bytes = dictionary[code]
    decompressed_data = list(current_bytes)

    for high_byte, low_byte in zip(compressed_data, compressed_data):
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
    decompressed_array = decompressed_array.reshape(shape)

    return decompressed_array


def binary_to_image(hidden_img_binary, shape):
    # Convert the binary array to bytes
    hidden_img_bytes = np.packbits(hidden_img_binary)

    # Convert the byte array to a numpy array
    hidden_img_array = np.frombuffer(hidden_img_bytes, dtype=np.uint8)

    # Reshape the array into an image
    hidden_img_array = hidden_img_array.reshape(shape)

    return hidden_img_array, shape


def embed(hidden_img_path, secret_key):
    print("====================== Embedding Process ======================")

    # FIXME: Remove the conversion of carrier image to grayscale
    hidden_img = cv2.imread(hidden_img_path, cv2.IMREAD_GRAYSCALE)

    print("Original Hidden Image - Total Pixel Size - ", (hidden_img.shape[0] * hidden_img.shape[1]))

    secret_key = adjust_key_length(secret_key)

    # FIXME: Encrypt the hidden image
    encrypted_hidden_img, original_shape = encrypt_image(hidden_img, secret_key)

    print("Encrypted Hidden Image - Total Pixel Size - ", len(encrypted_hidden_img))

    # FIXME: Compress the encrypted image using LZW Compression
    compressed_hidden_img, original_shape = lzw_compress((encrypted_hidden_img, original_shape))

    print("Compressed Hidden Image - Total Pixel Size - ", len(compressed_hidden_img))

    # Step 2: Convert the hidden image to a binary stream
    compressed_hidden_img_uint8 = np.array(compressed_hidden_img, dtype=np.uint8)
    compressed_hidden_img_binary = np.unpackbits(compressed_hidden_img_uint8)

    print("===============================================================")

    # Total number of bits of hidden image
    hidden_img_total_bits = len(compressed_hidden_img) * 8

    # TODO:Save the position sequences to a txt file
    with open('position_sequences.txt', 'w') as f:
        # Write the number of rows and columns to the first line
        f.write(f'{hidden_img.shape[0]} {hidden_img.shape[1]}\n')
        f.write(f'{len(encrypted_hidden_img)}\n')
        f.write(f'{len(compressed_hidden_img)}\n')

    print("------------------------------------------------------------------------")
    print("\t\tEmbedding Process")
    print("------------------------------------------------------------------------")
    print("The carrier image is 8-bit grayscale")
    print("1 pixel is equal to 8 bits")
    print("In 8 bits only the least significant bit is used")
    print("Available Total Number of Bits = (height * width) - total number bits of hidden image")
    print("------------------------------------------------------------------------")

    print(f"Total Number of Bits in the Hidden Image - {hidden_img_total_bits}")

    print("\nSuccessfully embedded the hidden image")

    print("------------------------------------------------------------------------")

    # Total number of bits of hidden image
    hidden_img_total_bits = len(compressed_hidden_img) * 8

    # Divide the hidden binary length to 8 for rgb image that has 8 bits per pixel
    compressed_hidden_img_binary = [bin(b)[2:].zfill(8) for b in compressed_hidden_img]

    # TODO: Save the position sequences to a txt file
    with open('position_sequences.txt', 'w') as f:
        # Write the number of rows and columns to the first line
        f.write(f'{hidden_img.shape[0]} {hidden_img.shape[1]}\n')
        f.write(f'{len(encrypted_hidden_img)}\n')
        f.write(f'{len(compressed_hidden_img)}\n')

    # TODO: Embed the hidden image into the carrier image using 3-2-3 technique
    for i in range(0, len(compressed_hidden_img_binary)):
        bits = compressed_hidden_img_binary[i]

    print("------------------------------------------------------------------------")
    print("\t\tEmbedding Process")
    print("------------------------------------------------------------------------")
    print("The carrier image is 24-bit RGB")
    print("1 pixel is equal to 24 bits")
    print("In a 24-bit image, only 8 bits are used using the 3-2-3 technique")
    print("Available Total Number of Bits = (height * width * 8) - total number bits of hidden image")
    print("------------------------------------------------------------------------")

    print(f"Total Number of Bits in the Hidden Image - {hidden_img_total_bits}")

    print("\nSuccessfully embedded the hidden image")

    print("------------------------------------------------------------------------")


def main():
    hidden_img_path = input("Enter path to hidden image: ")
    secret_key = input("Enter secret key: ")
    embed(hidden_img_path, secret_key)


if __name__ == "__main__":
    main()

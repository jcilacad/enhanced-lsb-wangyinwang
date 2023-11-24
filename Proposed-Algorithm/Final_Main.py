import numpy as np
import cv2
import random
import itertools

from PIL import Image
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


def embed(carrier_img_path, hidden_img_path, secret_key):
    # FIXME: Remove the conversion of carrier image to grayscale
    carrier_img = cv2.imread(carrier_img_path, cv2.IMREAD_COLOR)
    hidden_img = cv2.imread(hidden_img_path, cv2.IMREAD_GRAYSCALE)

    print("Original Hidden Image - Total Pixel Size - ", (hidden_img.shape[0] * hidden_img.shape[1]))

    secret_key = adjust_key_length(secret_key)

    # FIXME: Encrypt the hidden image
    encrypted_hidden_img, original_shape = encrypt_image(hidden_img, secret_key)

    print("Encrypted Hidden Image - Total Pixel Size - ", len(encrypted_hidden_img))

    # FIXME: Compress the encrypted image using LZW Compression
    compressed_hidden_img, original_shape = lzw_compress((encrypted_hidden_img, original_shape))

    print("Compressed Hidden Image - Total Pixel Size - ", len(compressed_hidden_img))

    # TODO: Use this for compression process

    decompressed_hidden_img = lzw_decompress((compressed_hidden_img, len(encrypted_hidden_img)))

    print("Decompressed Hidden Image - Total Pixel Size - ", len(decompressed_hidden_img))

    # TODO: Use this for decryption process
    decrypted_img, decrypted_shape = decrypt_image(decompressed_hidden_img, original_shape, secret_key)

    print("Decrypted Hidden Image - Total Pixel Size - ", (decrypted_img.shape[0] * decrypted_shape[1]))

    # FIXME: Ensure the compressed image is not larger than the carrier image
    assert len(compressed_hidden_img) <= carrier_img.size, "The hidden image is larger than the carrier image"

    # Step 2: Convert the hidden image to a binary stream
    compressed_hidden_img_uint8 = np.array(compressed_hidden_img, dtype=np.uint8)
    compressed_hidden_img_binary = np.unpackbits(compressed_hidden_img_uint8)

    # Steps 3 & 4: Generate a list of all pixel coordinates in the carrier image
    pixel_coords = [(i, j) for i in range(carrier_img.shape[0]) for j in range(carrier_img.shape[1])]

    # Step 5: Generate a random sequence of list from pixel_coords
    random_pixel_coords = random.sample(pixel_coords, len(pixel_coords))

    # Pixels to embed from hidden image to carrier image
    sliced_pixel_coords = itertools.islice(random_pixel_coords, len(compressed_hidden_img_binary))

    # TODO: Add the len(encrypted_hidden_img) to the 2nd line of txt file for decompression purposes
    # Save the position sequences to a txt file
    with open('position_sequences.txt', 'w') as f:
        # Write the number of rows and columns to the first line
        f.write(f'{hidden_img.shape[0]} {hidden_img.shape[1]}\n')
        f.write(f'{len(encrypted_hidden_img)}\n')
        f.write(f'{len(compressed_hidden_img)}\n')

        # Write the position sequences
        for pos in sliced_pixel_coords:
            f.write(f'{pos[0]} {pos[1]}\n')

    # TODO: Check if the image is grayscale or RGB
    if carrier_img is not None:
        if len(carrier_img.shape) == 2 or (
                len(carrier_img.shape) == 3 and np.all(carrier_img[:, :, 0] == carrier_img[:, :, 1]) and np.all(
            carrier_img[:, :, 0] == carrier_img[:, :, 2])):

            print("The image is 8-bit grayscale.")
            for bit, pos in zip(compressed_hidden_img_binary,
                                itertools.islice(random_pixel_coords, len(compressed_hidden_img_binary))):
                carrier_img[pos] = (carrier_img[pos] & ~1) | bit

            # Save the stego-image
            cv2.imwrite('stego_image.png', carrier_img)
        elif len(carrier_img.shape) == 3:

            print("The image is 24-bit RGB.")

            # Embed the hidden image into the carrier image
            # for i in range(0, len(compressed_img_bin), 8):
            #     bits = compressed_img_bin[i:i + 8]
            #     red_bits, blue_bits, green_bits = bits[:3], bits[3:5], bits[5:]
            #     red = int(''.join(map(str, red_bits)), 2)
            #     blue = int(''.join(map(str, blue_bits)), 2)
            #     green = int(''.join(map(str, green_bits)), 2)
            #     pos = random_pixel_coords[i // 8]
            #     carrier_img[pos] = (carrier_img[pos] & np.array([~7, ~3, ~7])) | np.array([red, blue, green])
            #
            # # Save the stego-image
            # cv2.imwrite('stego_image.png', carrier_img)
        else:
            print("The image is neither 8-bit grayscale nor 24-bit RGB.")
            raise ValueError("The image is neither 8-bit grayscale nor 24-bit RGB.")
    else:
        print("The image could not be read.")


def extract(stego_img_path, position_sequences_path, secret_key):
    # TODO: Use this for compression process
    # decompressed_hidden_img = lzw_decompress((compressed_hidden_img,
    #                                           (len(encrypted_hidden_img))))
    #
    # print("Decompressed Hidden Image - Total Pixel Size - ", len(decompressed_hidden_img))

    # TODO: Use this for decryption process
    # decrypted_img, decrypted_shape = decrypt_image(encrypted_hidden_img, original_shape, secret_key)
    #
    # print(decrypted_img)

    #
    # bit_image = np.unpackbits(decrypted_img)
    # print(bit_image)
    # print(decrypted_img)

    # Step 1:
    # Load the stego image
    stego_img = cv2.imread(stego_img_path, cv2.IMREAD_COLOR)

    # Step 2: Load the position sequences from the text file
    with open(position_sequences_path, 'r') as f:
        # Read the first line and split it into rows and columns
        rows, cols = map(int, next(f).strip().split())

        # Read the second line
        second_line = next(f).strip()

        # Convert it to an integer
        encrypted_img_length = int(second_line)

        # Read the third line
        third_line = next(f).strip()

        # Split the line into parts and convert each part to an integer
        compressed_img_length = int(third_line)

        # Read the remaining lines as position sequences
        position_sequences = [tuple(map(int, line.strip().split())) for line in f]

    # Check if the stego image is 8-bit grayscale or 24-bit rgb image
    if stego_img is not None:
        if len(stego_img.shape) == 2 or (
                len(stego_img.shape) == 3 and np.all(stego_img[:, :, 0] == stego_img[:, :, 1]) and np.all(
            stego_img[:, :, 0] == stego_img[:, :, 2])):

            print("The image is 8-bit grayscale.")

            stego_img = cv2.imread(stego_img_path, cv2.IMREAD_GRAYSCALE)

            # Step 3:
            # Extract the binary digital stream of the hidden image
            hidden_img_binary = np.array([stego_img[pos] & 1 for pos in position_sequences])

            # Ensure that the length of hidden_img_binary is a multiple of 8
            if hidden_img_binary.size % 8 != 0:
                padding = np.zeros(8 - hidden_img_binary.size % 8, dtype=np.uint8)
                hidden_img_binary = np.concatenate((hidden_img_binary, padding))

            # TODO: Convert the binary to image
            hidden_img_binary_array = binary_to_image(hidden_img_binary, compressed_img_length)
            print("Retrieved Compressed Hidden Message - Total Pixel Size - ", len(hidden_img_binary_array[0]))

            # TODO: Decompress hidden image using LZW compression
            decompressed_hidden_img = lzw_decompress((hidden_img_binary_array[0], encrypted_img_length))

            print("Decompressed Hidden Image - Total Pixel Size - ", len(decompressed_hidden_img))

            



            # # TODO: Use this for decryption process
            # decrypted_img, decrypted_shape = decrypt_image(decompressed_hidden_img, original_shape, secret_key)

            # Convert the binary array to bytes
            # hidden_img_bytes = np.packbits(hidden_img_binary)
            #
            # # Convert the byte array to a numpy array
            # hidden_img_array = np.frombuffer(hidden_img_bytes, dtype=np.uint8)
            #
            # # Reshape the array into an image (replace 'width' and 'height' with the actual dimensions of your image)
            # hidden_img_array = hidden_img_array.reshape((compressed_img_length, 1))
            #
            # # Create a PIL image from the array
            # hidden_img = Image.fromarray(hidden_img_array, 'L')
            #
            # print(hidden_img_binary)
            # print(len(hidden_img_binary))
            # Convert hidden image binary to bytes

            # decompressed_hidden_img = lzw_decompress((hidden_img_binary, encrypted_img_length))
            #
            # print(len(decompressed_hidden_img))





        elif len(stego_img.shape) == 3:

            print("The image is 24-bit RGB.")


        else:

            print("The image is neither 8-bit grayscale nor 24-bit RGB.")

    else:

        print("The image could not be read.")


def main():
    operation = input("Choose operation ('embed' or 'extract'): ")
    if operation == 'embed':
        carrier_img_path = input("Enter path to carrier image: ")
        hidden_img_path = input("Enter path to hidden image: ")
        secret_key = input("Enter secret key: ")
        embed(carrier_img_path, hidden_img_path, secret_key)
    elif operation == 'extract':
        stego_img_path = input("Enter path to stego-image: ")
        position_sequences_path = input("Enter path to position sequences txt file: ")
        secret_key = input("Enter secret key: ")
        extract(stego_img_path, position_sequences_path, secret_key)


if __name__ == "__main__":
    main()

import numpy as np
import cv2
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


# Encrypt hidden image
def encrypt_image(img, key):
    img_data = img.tobytes()

    cipher = AES.new(key, AES.MODE_ECB)
    encrypted_data = cipher.encrypt(pad(img_data, AES.block_size))

    encrypted_img = np.frombuffer(encrypted_data, dtype=np.uint8)

    return encrypted_img, img.shape


# Decrypt hidden image
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
    compressed_data = []

    for byte in input_bytes:
        new_bytes = current_bytes + bytes([byte])
        if new_bytes in dictionary:
            current_bytes = new_bytes
        else:
            compressed_data.append(dictionary[current_bytes])
            dictionary[new_bytes] = len(dictionary)
            current_bytes = bytes([byte])

    if current_bytes:
        compressed_data.append(dictionary[current_bytes])

    return compressed_data, shape


def lzw_decompress(input_tuple):
    compressed_data, shape = input_tuple
    dictionary = {i: bytes([i]) for i in range(256)}
    current_bytes = bytes([compressed_data[0]])
    decompressed_data = list(current_bytes)

    for code in compressed_data[1:]:
        if code not in dictionary:
            new_bytes = current_bytes + current_bytes[:1]
        else:
            new_bytes = dictionary[code]

        decompressed_data.extend(new_bytes)
        dictionary[len(dictionary)] = current_bytes + new_bytes[:1]
        current_bytes = new_bytes

    decompressed_array = np.frombuffer(bytes(decompressed_data), dtype=np.uint8)
    decompressed_array = decompressed_array.reshape(shape)

    return decompressed_array


# Embedding Process
def embed(carrier_img_path, hidden_img_path, secret_key):
    # FIXME: Remove the conversion of carrier image to grayscale
    carrier_img = cv2.imread(carrier_img_path, cv2.IMREAD_COLOR)
    hidden_img = cv2.imread(hidden_img_path, cv2.IMREAD_GRAYSCALE)

    print("Original Hidden Image - Total Pixel Size - ", (hidden_img.shape[0] * hidden_img.shape[1]))

    # Adjust the key length to 16
    secret_key = adjust_key_length(secret_key)

    # FIXME: Encrypt the hidden image
    encrypted_hidden_img, original_shape = encrypt_image(hidden_img, secret_key)

    print("Encrypted Hidden Image - Total Pixel Size - ", len(encrypted_hidden_img))

    # FIXME: Compress the encrypted image using LZW Compression
    compressed_hidden_img, original_shape = lzw_compress((encrypted_hidden_img, original_shape))

    print("Compressed Hidden Image - Total Pixel Size - ", len(compressed_hidden_img))



    # FIXME: Ensure the compressed image is not larger than the carrier image
    assert len(compressed_hidden_img) <= carrier_img.size, "The hidden image is larger than the carrier image"


    # TODO: Add the len(encrypted_hidden_img) to the 2nd line of txt file for compression purposes
    #
    # # Step 2:
    # # Convert the hidden image to a binary stream
    # hidden_img_binary = np.unpackbits(hidden_img)
    #
    # # Steps 3 & 4:
    # # Generate a list of all pixel coordinates in the carrier image
    # pixel_coords = [(i, j) for i in range(carrier_img.shape[0]) for j in range(carrier_img.shape[1])]
    #
    # # Step 5:
    # # Generate a random sequence of list from pixel_coords
    # random_pixel_coords = random.sample(pixel_coords, len(pixel_coords))
    #
    # # Pixels to embed from hidden image to carrier image
    # sliced_pixel_coords = itertools.islice(random_pixel_coords, len(hidden_img_binary))
    #
    # # Save the position sequences to a txt file
    # with open('position_sequences.txt', 'w') as f:
    #     # Write the number of rows and columns to the first line
    #     f.write(f'{hidden_img.shape[0]} {hidden_img.shape[1]}\n')
    #
    #     # Write the position sequences
    #     for pos in sliced_pixel_coords:
    #         f.write(f'{pos[0]} {pos[1]}\n')
    #
    # # Step 6:
    # # Embed the hidden image into the carrier image
    # for bit, pos in zip(hidden_img_binary, itertools.islice(random_pixel_coords, len(hidden_img_binary))):
    #     carrier_img[pos] = (carrier_img[pos] & ~1) | bit
    #
    # # Save the stego-image
    # cv2.imwrite('stego_image.png', carrier_img)


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
    cv2.imwrite('hidden_image.png', hidden_img)


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

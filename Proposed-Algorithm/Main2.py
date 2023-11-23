import ncompress
import numpy as np
import cv2
import random
import itertools
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.Util.Padding import pad, unpad

def embed(carrier_img_path, hidden_img_path, encryption_key):
    # Step 1:
    # Read the images

    # Remove the conversion of carrier image to grayscale
    # Read the carrier image in color mode
    carrier_img = cv2.imread(carrier_img_path, cv2.IMREAD_COLOR)
    hidden_img = cv2.imread(hidden_img_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the hidden image is not larger than the carrier image
    assert hidden_img.size <= carrier_img.size, "The hidden image is larger than the carrier image"

    # Step 2:
    # Convert the hidden image to a binary stream
    hidden_img_binary = np.unpackbits(hidden_img)

    # Convert hidden_img_binary to bytes for encryption and compression
    hidden_img_bytes = hidden_img.tobytes()

    # Use SHA-256 to generate a 16-byte key, then truncate to 16 bytes for AES-128
    key = SHA256.new(encryption_key.encode()).digest()[:16]

    # Create a new AES cipher object with the hashed key and AES.MODE_CBC mode
    cipher_encrypt = AES.new(key, AES.MODE_CBC)

    # Pad the data to a multiple of 16 bytes before encryption
    padded_img_bytes = pad(hidden_img_bytes, 16)

    print(f'Padded data: {padded_img_bytes}')  # Debugging line

    # Encrypt the padded data
    encrypted_img_bytes = cipher_encrypt.encrypt(padded_img_bytes)

    print(len(encrypted_img_bytes))

    # Create a new AES cipher object for decryption
    cipher_decrypt = AES.new(key, AES.MODE_CBC, iv=cipher_encrypt.iv)

    # Decrypt the encrypted bytes
    decrypted_img_bytes = cipher_decrypt.decrypt(encrypted_img_bytes)

    # Unpad the decrypted bytes
    unpadded_img_bytes = unpad(decrypted_img_bytes, 16)

    print(len(unpadded_img_bytes))


    # # Convert ct_bytes to a binary stream
    # ct_bytes_binary = np.unpackbits(np.frombuffer(encrypted_img_bytes, dtype=np.uint8))
    #
    # # Compress the encrypted bytes
    # compressed_img_binary = lzw_compress(ct_bytes_binary)
    #
    # # Convert bytes to binary
    # encrypted_img_bin = ''.join(format(byte, '08b') for byte in encrypted_img_bytes)
    #
    # print("============================ original")
    # # print(hidden_img_binary)
    # print(len(hidden_img_binary))
    #
    # print("============================ encrypted")
    # # print(encrypted_img_bin)
    # print(len(encrypted_img_bin))
    #
    # print("============================ Compressed")
    # # print(compressed_img_bin)
    # print(len(compressed_img_binary))
    #
    # # Steps 3 & 4:
    # # Generate a list of all pixel coordinates in the carrier image
    # pixel_coords = [(i, j) for i in range(carrier_img.shape[0]) for j in range(carrier_img.shape[1])]
    #
    # # Step 5:
    # # Generate a random sequence of list from pixel_coords
    # random_pixel_coords = random.sample(pixel_coords, len(pixel_coords))
    #
    # # TODO: Create a condition for RGB Image, Using 3-2-3 technique, for 8 bits we are only going to embed that in one pixel
    #
    # # Pixels to embed from hidden image to carrier image
    # sliced_pixel_coords = itertools.islice(random_pixel_coords, len(compressed_img_binary))
    #
    #
    # ## Error begins here
    #
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
    #
    # ## TODO: Up to here
    #
    #
    # # Step 6:
    # # Embed the hidden image into the carrier image
    #
    # # Check if the image is grayscale or RGB
    # if carrier_img is not None:
    #     if len(carrier_img.shape) == 2 or (
    #         len(carrier_img.shape) == 3 and np.all(carrier_img[:, :, 0] == carrier_img[:, :, 1]) and np.all(
    #         carrier_img[:, :, 0] == carrier_img[:, :, 2])):
    #
    #         print("The image is 8-bit grayscale.")
    #         for bit, pos in zip(compressed_img_bin, itertools.islice(random_pixel_coords, len(compressed_img_bin))):
    #             carrier_img[pos] = (carrier_img[pos] & ~1) | bit
    #
    #         # Save the stego-image
    #         cv2.imwrite('stego_image.png', carrier_img)
    #     elif len(carrier_img.shape) == 3:
    #
    #         print("The image is 24-bit RGB.")
    #
    #         # Embed the hidden image into the carrier image
    #         for i in range(0, len(compressed_img_bin), 8):
    #             bits = compressed_img_bin[i:i + 8]
    #             red_bits, blue_bits, green_bits = bits[:3], bits[3:5], bits[5:]
    #             red = int(''.join(map(str, red_bits)), 2)
    #             blue = int(''.join(map(str, blue_bits)), 2)
    #             green = int(''.join(map(str, green_bits)), 2)
    #             pos = random_pixel_coords[i // 8]
    #             carrier_img[pos] = (carrier_img[pos] & np.array([~7, ~3, ~7])) | np.array([red, blue, green])
    #
    #         # Save the stego-image
    #         cv2.imwrite('stego_image.png', carrier_img)
    #     else:
    #         print("The image is neither 8-bit grayscale nor 24-bit RGB.")
    #         raise ValueError("The image is neither 8-bit grayscale nor 24-bit RGB.")
    # else:
    #     print("The image could not be read.")


def extract(stego_img_path, position_sequences_path, encryption_key):


    # Step 1:
    # Load the stego image
    stego_img = cv2.imread(stego_img_path, cv2.IMREAD_COLOR)

    # Step 2:
    # Load the position sequences from the text file
    with open(position_sequences_path, 'r') as f:
        # Read the first line and split it into rows and columns
        rows, cols = map(int, next(f).strip().split())

        # Read the remaining lines as position sequences
        position_sequences = [tuple(map(int, line.strip().split())) for line in f]
        # print("Position Sequences", position_sequences)



    # Check if the stego image is 8-bit grayscale or 24-bit rgb image
    if stego_img is not None:
        if len(stego_img.shape) == 2 or (
            len(stego_img.shape) == 3 and np.all(stego_img[:, :, 0] == stego_img[:, :, 1]) and np.all(
            stego_img[:, :, 0] == stego_img[:, :, 2])):

            print("The image is 8-bit grayscale.")

            # Step 3:
            # Extract the binary digital stream of the hidden image
            hidden_img_binary = np.array([stego_img[pos] & 1 for pos in position_sequences])

            # Ensure that the length of hidden_img_binary is a multiple of 8
            if hidden_img_binary.size % 8 != 0:
                padding = np.zeros(8 - hidden_img_binary.size % 8, dtype=np.uint8)
                hidden_img_binary = np.concatenate((hidden_img_binary, padding))

            # Convert hidden image binary to bytes
            hidden_img_bytes = hidden_img_binary.tobytes()

            # Decompress hidden img bytes
            decompressed_img_bytes = lzw_decompress(hidden_img_bytes)

            key = SHA256.new(encryption_key.encode()).digest()[:16]

            # Create a new AES cipher object with the hashed key
            cipher = AES.new(key, AES.MODE_CBC)

            # Decrypt the decompressed bytes
            decrypted_img_bytes = cipher.decrypt(decompressed_img_bytes)

            # Unpad the decrypted bytes
            unpadded_img_bytes = unpad(decrypted_img_bytes, 16)

            decrypted_img_bin_n_array = ''.join(format(byte, '08b') for byte in unpadded_img_bytes)

            # Convert the binary string into an array of individual bits
            decrypted_img_bin_array = [int(bit) for bit in decrypted_img_bin_n_array]

            # Convert list to numpy array
            hidden_img_binary = np.array(decrypted_img_bin_array)

            # Step 4:
            # Convert the binary digital stream back into pixel form
            hidden_img_pixels = np.packbits(hidden_img_binary)
            # Step 5:
            # Reshape the pixel data to form the hidden image
            hidden_img = np.reshape(hidden_img_pixels, (rows, cols))

            # Save the hidden image in the root directory of the project
            cv2.imwrite('hidden_image.png', hidden_img)

        elif len(stego_img.shape) == 3:

            print("The image is 24-bit RGB.")
            with open('position_sequences.txt', 'r') as f:
                hidden_img_shape = tuple(map(int, f.readline().split()))
                pixel_coords = [tuple(map(int, line.split())) for line in f]
            random_pixel_coords = random.sample(pixel_coords, len(pixel_coords))
            compressed_img_bin = ''
            for pos in itertools.islice(random_pixel_coords, np.prod(hidden_img_shape)):
                red, blue, green = stego_img[pos] & np.array([7, 3, 7])
                bits = list(map(int, format(red, '03b') + format(blue, '02b') + format(green, '03b')))
                compressed_img_bin += ''.join(map(str, bits))
            compressed_img_bytes = bytes(
                int(compressed_img_bin[i:i + 8], 2) for i in range(0, len(compressed_img_bin), 8))

            print("Len", len(compressed_img_bytes))
            encrypted_img_bytes = lzw_decompress(compressed_img_bytes)  # Use ncompress for decompression
            # Use SHA-256 to generate a 32-byte key, then truncate to 16 bytes for AES-128
            key = SHA256.new(encryption_key.encode()).digest()[:16]

            # Create a new AES cipher object with the hashed key
            cipher = AES.new(key, AES.MODE_ECB)

            # Decrypt the decompressed bytes
            decrypted_img_bytes = cipher.decrypt(encrypted_img_bytes)

            print(f'Decrypted data: {decrypted_img_bytes}')  # Debugging line

            # Unpad the decrypted bytes
            unpadded_img_bytes = unpad(decrypted_img_bytes, AES.block_size)

            # Convert the unpadded bytes back to binary
            hidden_img_binary = np.frombuffer(unpadded_img_bytes, dtype=np.uint8)

            # Reshape the binary data back to the original image shape
            hidden_img = np.packbits(hidden_img_binary).reshape(hidden_img_shape)

            cv2.imwrite('hidden_image.png', hidden_img)

        else:
            print("The image is neither 8-bit grayscale nor 24-bit RGB.")

    else:
        print("The image could not be read.")


def lzw_compress(input_binary):
    # Convert the binary stream to bytes
    input_bytes = np.packbits(input_binary)

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

    # Convert the list of codes into bytes
    compressed_bytes = bytearray()
    for code in compressed_data:
        compressed_bytes.extend(code.to_bytes((code.bit_length() + 7) // 8, 'big'))

    # Convert the compressed bytes back to a binary stream
    compressed_binary = np.unpackbits(np.frombuffer(compressed_bytes, dtype=np.uint8))

    return compressed_binary

def lzw_decompress(compressed_bytes):
    dictionary = {i: bytes([i]) for i in range(256)}
    current_bytes = dictionary[compressed_bytes[0]]
    decompressed_data = [current_bytes]

    for code in compressed_bytes[1:]:
        if code not in dictionary:
            new_bytes = current_bytes + current_bytes[:1]
        else:
            new_bytes = dictionary[code]

        decompressed_data.append(new_bytes)

        dictionary[len(dictionary)] = current_bytes + new_bytes[:1]
        current_bytes = new_bytes

    return b"".join(decompressed_data)

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

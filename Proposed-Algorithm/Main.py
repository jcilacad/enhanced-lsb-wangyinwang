import ncompress
import numpy as np
import cv2
import random
import itertools
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.Util.Padding import pad


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
    # Convert the hidden image to a binary stream and encrypt it.
    hidden_img_binary = np.unpackbits(hidden_img)

    # Convert hidden_img_binary to bytes for encryption and compression
    hidden_img_bytes = hidden_img_binary.tobytes()

    # Use SHA-256 to generate a 32-byte key
    key = SHA256.new(encryption_key.encode()).digest()

    # Create a new AES cipher object with the hashed key
    cipher = AES.new(key, AES.MODE_ECB)

    # Encrypt the data
    encrypted_img_bytes = cipher.encrypt(pad(hidden_img_bytes, AES.block_size))

    # Compress the encrypted bytes
    compressed_img_bytes = ncompress.compress(encrypted_img_bytes)

    # Convert bytes to binary
    encrypted_img_bin = ''.join(format(byte, '08b') for byte in encrypted_img_bytes)
    compressed_img_bin = ''.join(format(byte, '08b') for byte in compressed_img_bytes)

    print("============================ original")
    # print(hidden_img_binary)
    print(len(hidden_img_binary))

    print("============================ encrypted")
    # print(encrypted_img_bin)
    print(len(encrypted_img_bin))

    print("============================ Compressed")
    # print(compressed_img_bin)
    print(len(compressed_img_bin))

    # Steps 3 & 4:
    # Generate a list of all pixel coordinates in the carrier image
    pixel_coords = [(i, j) for i in range(carrier_img.shape[0]) for j in range(carrier_img.shape[1])]

    # Step 5:
    # Generate a random sequence of list from pixel_coords
    random_pixel_coords = random.sample(pixel_coords, len(pixel_coords))

    # Pixels to embed from hidden image to carrier image
    sliced_pixel_coords = itertools.islice(random_pixel_coords, len(compressed_img_bin))

    # Save the position sequences to a txt file
    with open('position_sequences.txt', 'w') as f:
        # Write the number of rows and columns to the first line
        f.write(f'{hidden_img.shape[0]} {hidden_img.shape[1]}\n')

        # Write the position sequences
        for pos in sliced_pixel_coords:
            f.write(f'{pos[0]} {pos[1]}\n')

    # Step 6:
    # Embed the hidden image into the carrier image

    # Check if the image is grayscale or RGB
    if carrier_img is not None:
        if len(carrier_img.shape) == 2 or (
            len(carrier_img.shape) == 3 and np.all(carrier_img[:, :, 0] == carrier_img[:, :, 1]) and np.all(
            carrier_img[:, :, 0] == carrier_img[:, :, 2])):
            print("The image is 8-bit grayscale.")
            for bit, pos in zip(compressed_img_bin, itertools.islice(random_pixel_coords, len(compressed_img_bin))):
                carrier_img[pos][0] = (carrier_img[pos][0] & ~1) | int(bit)

            # Save the stego-image
            cv2.imwrite('stego_image.png', carrier_img)
        elif len(carrier_img.shape) == 3:

            print("The image is 24-bit RGB.")

            # Create an iterator for the compressed_data
            compressed_data_iter = iter(compressed_img_bin)

            for pos in itertools.islice(random_pixel_coords, len(compressed_img_bin) // 8):
                # Get the next 8 bits from compressed_data
                bits = [next(compressed_data_iter) for _ in range(8)]

                # Split the bits among the color channels
                red_bits, blue_bits, green_bits = bits[:3], bits[3:5], bits[5:]

                # Convert the bits to integers
                red = int(''.join(map(str, red_bits)), 2)
                blue = int(''.join(map(str, blue_bits)), 2)
                green = int(''.join(map(str, green_bits)), 2)

                # Embed the bits in the color channels of the image
                carrier_img[pos] = (carrier_img[pos] & np.array([~7, ~3, ~7])) | np.array([red, blue, green])

            # Save the stego-image
            cv2.imwrite('stego_image.png', carrier_img)
        else:
            print("The image is neither 8-bit grayscale nor 24-bit RGB.")
            raise ValueError("The image is neither 8-bit grayscale nor 24-bit RGB.")
    else:
        print("The image could not be read.")


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
    cv2.imwrite('hidden_image.png', hidden_img)



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

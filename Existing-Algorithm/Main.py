import numpy as np
import cv2
import random
import itertools

def embed(carrier_img_path, hidden_img_path):

    # Step 1:
    # Read the images
    carrier_img = cv2.imread(carrier_img_path, cv2.IMREAD_GRAYSCALE)
    hidden_img = cv2.imread(hidden_img_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the hidden image is not larger than the carrier image
    assert hidden_img.size <= carrier_img.size, "The hidden image is larger than the carrier image"

    # Step 2:
    # Convert the hidden image to a binary stream
    hidden_img_binary = np.unpackbits(hidden_img)

    # Steps 3 & 4:
    # Generate a list of all pixel coordinates in the carrier image
    pixel_coords = [(i, j) for i in range(carrier_img.shape[0]) for j in range(carrier_img.shape[1])]

    # Step 5:
    # Generate a random sequence of list from pixel_coords
    random_pixel_coords = random.sample(pixel_coords, len(pixel_coords))

    print(hidden_img_binary)

    # Pixels to embed from hidden image to carrier image
    sliced_pixel_coords = itertools.islice(random_pixel_coords, len(hidden_img_binary))

    # Save the position sequences to a txt file
    with open('position_sequences.txt', 'w') as f:
        # Write the number of rows and columns to the first line
        f.write(f'{hidden_img.shape[0]} {hidden_img.shape[1]}\n')

        # Write the position sequences
        for pos in sliced_pixel_coords:
            f.write(f'{pos[0]} {pos[1]}\n')

    # Step 6:
    # Embed the hidden image into the carrier image
    for bit, pos in zip(hidden_img_binary, itertools.islice(random_pixel_coords, len(hidden_img_binary))):
        carrier_img[pos] = (carrier_img[pos] & ~1) | bit

    # Save the stego-image
    cv2.imwrite('stego_image.png', carrier_img)


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
        embed(carrier_img_path, hidden_img_path)
    elif operation == 'extract':
        stego_img_path = input("Enter path to stego-image: ")
        position_sequences_path = input("Enter path to position sequences txt file: ")
        extract(stego_img_path, position_sequences_path)

if __name__ == "__main__":
    main()

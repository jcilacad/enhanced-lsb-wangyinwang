import numpy as np
import cv2
import random

def embed(carrier_img_path, hidden_img_path):

    # Step 1:
    # Read the images
    carrier_img = cv2.imread(carrier_img_path, cv2.IMREAD_GRAYSCALE)
    hidden_img = cv2.imread(hidden_img_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the hidden image is not larger than the carrier image
    assert hidden_img.size <= carrier_img.size, "The hidden image is larger than the carrier image"

    # Save the shape of the hidden image
    with open('hidden_img_shape.txt', 'w') as f:
        f.write(f'{hidden_img.shape[0]} {hidden_img.shape[1]}')

    # Step 2:
    # Convert the hidden image to a binary stream
    hidden_img_binary = np.unpackbits(hidden_img)

    # Steps 3 & 4:
    # Generate a list of all pixel coordinates in the carrier image
    pixel_coords = [(i, j) for i in range(carrier_img.shape[0]) for j in range(carrier_img.shape[1])]

    # Step 5:
    # Generate a random sequence of list from pixel_coords
    random_pixel_coords = random.sample(pixel_coords, len(pixel_coords))

    # Save the position sequences to a txt file
    with open('position_sequences.txt', 'w') as f:
        for pos in random_pixel_coords:
            f.write(f'{pos[0]} {pos[1]}\n')

    # Step 6:
    # Embed the hidden image into the carrier image
    for bit, pos in zip(hidden_img_binary, random_pixel_coords):
        carrier_img[pos] = (carrier_img[pos] & ~1) | bit

    # Save the stego-image
    cv2.imwrite('stego_image.png', carrier_img)

def extract(stego_img_path, position_sequences_path):

    # Step 1:
    # Load the stego image and convert it to grayscale
    stego_img = cv2.imread(stego_img_path, cv2.IMREAD_GRAYSCALE)

    rows, cols = stego_img.shape

    # Step 2:
    # Load the position sequences from the text file
    with open(position_sequences_path, 'r') as f:
        position_sequences = [tuple(map(int, line.strip().split())) for line in f]

    # Step 3:
    # Extract the binary digital stream of the hidden image
    hidden_img_binary = [stego_img[pos] & 1 for pos in position_sequences]

    # Load the shape of the hidden image
    with open('hidden_img_shape.txt', 'r') as f:
        hidden_img_shape = tuple(map(int, f.readline().strip().split()))

    # Step 4:
    # Convert the binary digital stream back into pixel form
    hidden_img_pixels = np.packbits(hidden_img_binary)

    # Total size of hidden image pixels
    total_size = hidden_img_shape[0] * hidden_img_shape[1];



    # Step 5:
    # Reshape the pixel data to form the hidden image
    hidden_img = np.reshape(hidden_img_pixels[:total_size], hidden_img_shape)

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

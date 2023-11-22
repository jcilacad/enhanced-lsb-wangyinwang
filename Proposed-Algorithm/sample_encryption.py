from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

def encrypt_image(image_bytes, key):
    # Create a new AES cipher object
    cipher = AES.new(key, AES.MODE_CBC)

    # Encrypt the data
    encrypted_data = cipher.encrypt(pad(image_bytes, AES.block_size))

    # Return the IV and the encrypted data
    return cipher.iv + encrypted_data

def decrypt_image(encrypted_data, key):
    # Extract the IV from the first 16 bytes
    iv = encrypted_data[:16]
    encrypted_data = encrypted_data[16:]

    # Create a new AES cipher object
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)

    # Decrypt the data
    decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)

    # Return the decrypted data
    return decrypted_data

def main():
    # Prompt the user to enter the image bytes and the key
    image_bytes = bytes.fromhex(input("Enter the image bytes (in hexadecimal): "))
    key = bytes.fromhex(input("Enter the key (16 bytes, in hexadecimal): "))

    # Prompt the user to choose between encryption and decryption
    choice = input("Enter 1 to encrypt, 2 to decrypt: ")

    if choice == '1':
        # Encrypt the image
        encrypted_data = encrypt_image(image_bytes, key)
        print("Encrypted data (in hexadecimal):", encrypted_data.hex())
    elif choice == '2':
        # Decrypt the image
        decrypted_data = decrypt_image(image_bytes, key)
        print("Decrypted data (in hexadecimal):", decrypted_data.hex())
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
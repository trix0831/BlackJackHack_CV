import serial
import time

# Configure the serial connection
arduino = serial.Serial(port='COM11', baudrate=9600, timeout=1)  # Adjust 'COM3' to your Arduino port

def send_command(command):
    arduino.write(command.encode())  # Send the command as bytes
    time.sleep(0.1)                  # Give the Arduino time to process

try:
    while True:
        user_input = input("Enter '1' to turn ON, '0' to turn OFF, or 'q' to quit: ").strip()
        if user_input == '1':
            send_command('1')
            print("LED turned ON")
        elif user_input == '0':
            send_command('0')
            print("LED turned OFF")
        elif user_input == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid input. Please enter '1', '0', or 'q'.")
finally:
    arduino.close()  # Close the serial connection

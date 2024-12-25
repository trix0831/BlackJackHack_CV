import serial
import time

# Configure the serial connection
arduino = serial.Serial(port='COM12', baudrate=9600, timeout=1)  # Adjust 'COM3' to your Arduino port

def send_command(command):
    arduino.write(command.encode())  # Send the command as bytes
    time.sleep(0.1)                  # Give the Arduino time to process

try:
    print("Connected to Arduino")
    while True:
        # sequentially play
        send_command('001')
        time.sleep(0.5)
        send_command('010')
        time.sleep(0.5)
        send_command('100')
        time.sleep(0.5)
        send_command('110')
        time.sleep(0.5)
        send_command('101')
        time.sleep(0.5)
        send_command('011')
        time.sleep(0.5)
        send_command('111')
        time.sleep(0.5)
    # while True:
    #     user_input = input("Enter '1' to turn ON, '0' to turn OFF, or 'q' to quit: ").strip()
    #     if user_input == '1':
    #         send_command('100')
    #         print("LED turned ON")
    #     elif user_input == '0':
    #         send_command('010')
    #         print("LED turned OFF")
    #     elif user_input == 'q':
    #         print("Exiting...")
    #         break
    #     else:
    #         print("Invalid input. Please enter '1', '0', or 'q'.")
finally:
    arduino.close()  # Close the serial connection

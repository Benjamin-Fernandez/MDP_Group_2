import serial
import time

# Open both serial ports
rfcomm = serial.Serial('/dev/rfcomm0', baudrate=9600, timeout=0.1)
ttyacm = serial.Serial('/dev/ttyACM0', baudrate=115200, timeout=0.1)

try:
    while True:
        if rfcomm.in_waiting:
            data = rfcomm.read(rfcomm.in_waiting)
            print(f"From rfcomm0: {data}")
            ttyacm.write(data)

        # (Optional) echo back from ttyACM0 if needed
        # if ttyacm.in_waiting:
        #     response = ttyacm.read(ttyacm.in_waiting)
        #     rfcomm.write(response)

        time.sleep(0.01)

except KeyboardInterrupt:
    print("Stopped.")

finally:
    rfcomm.close()
    ttyacm.close()


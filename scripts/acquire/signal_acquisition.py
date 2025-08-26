import spidev
import time
import gpiod


class ADS1299Controller:
    def __init__(self, spi_bus, spi_device, cs_line=None):
        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.max_speed_hz = 4000000
        self.spi.lsbfirst = False
        self.spi.mode = 0b01
        self.spi.bits_per_word = 8
        self.cs_line = cs_line
        
    def send_command(self, command):
        send_data = [command]
        if self.cs_line:
            self.cs_line.set_value(0)
            self.spi.xfer(send_data)
            self.cs_line.set_value(1)
        else:
            self.spi.xfer(send_data)
    
    def write_byte(self, register, data):
        write = 0x40
        register_write = write | register
        data_array = [register_write, 0x00, data]
        
        if self.cs_line:
            self.cs_line.set_value(0)
            self.spi.xfer(data_array)
            self.cs_line.set_value(1)
        else:
            self.spi.xfer(data_array)
    
    def read_bytes(self, length):
        if self.cs_line:
            self.cs_line.set_value(0)
            output = self.spi.readbytes(length)
            self.cs_line.set_value(1)
        else:
            output = self.spi.readbytes(length)
        return output
    
    def initialize(self):
        # ADS1299 registers
        wakeup = 0x02
        stop = 0x0A
        reset = 0x06
        sdatac = 0x11
        rdatac = 0x10
        start = 0x08
        config1 = 0x01
        config2 = 0x02
        config3 = 0x03
        ch1set = 0x05
        ch2set = 0x06
        ch3set = 0x07
        ch4set = 0x08
        ch5set = 0x09
        ch6set = 0x0A
        ch7set = 0x0B
        ch8set = 0x0C
        
        self.send_command(wakeup)
        self.send_command(stop)
        self.send_command(reset)
        self.send_command(sdatac)

        self.write_byte(0x14, 0x80)
        self.write_byte(config1, 0x96)
        self.write_byte(config2, 0xD4)
        self.write_byte(config3, 0xFF)
        self.write_byte(0x04, 0x00)
        self.write_byte(0x0D, 0x00)
        self.write_byte(0x0E, 0x00)
        self.write_byte(0x0F, 0x00)
        self.write_byte(0x10, 0x00)
        self.write_byte(0x11, 0x00)
        self.write_byte(0x15, 0x20)
        self.write_byte(0x17, 0x00)
        self.write_byte(ch1set, 0x00)
        self.write_byte(ch2set, 0x00)
        self.write_byte(ch3set, 0x00)
        self.write_byte(ch4set, 0x00)
        self.write_byte(ch5set, 0x00)
        self.write_byte(ch6set, 0x00)
        self.write_byte(ch7set, 0x01)
        self.write_byte(ch8set, 0x01)

        self.send_command(rdatac)
        self.send_command(start)


class SignalAcquisition:
    def __init__(self):
        # GPIO setup
        self.chip = gpiod.chip("0")
        
        # CS pin for second ADS1299
        cs_pin = 19
        self.cs_line = self.chip.get_line(cs_pin)
        cs_line_out = gpiod.line_request()
        cs_line_out.consumer = "SPI_CS"
        cs_line_out.request_type = gpiod.line_request.DIRECTION_OUTPUT
        self.cs_line.request(cs_line_out)
        self.cs_line.set_value(1)
        
        # Button pin for DRDY
        button_pin = 26
        self.button_line = self.chip.get_line(button_pin)
        button_request = gpiod.line_request()
        button_request.consumer = "Button"
        button_request.request_type = gpiod.line_request.DIRECTION_INPUT
        self.button_line.request(button_request)
        
        # Initialize ADS1299 controllers
        self.ads1 = ADS1299Controller(0, 0)
        self.ads2 = ADS1299Controller(0, 1, self.cs_line)
        
        self.ads1.initialize()
        self.ads2.initialize()
        
        # Data validation
        self.data_test = 0x7FFFFF
        self.data_check = 0xFFFFFF
        self.last_valid_value = 5
        self.counter = 0
        
        self.test_drdy = 5
        self.is_running = False
    
    def _to_signed_24bit(self, msb, middle, lsb):
        combined = (msb << 16) | (middle << 8) | lsb
        if (msb & 0x80) != 0:
            combined -= 1 << 24
        return combined
    
    def _is_valid_input(self, input_list):
        msb = input_list[24]
        middle = input_list[25]
        lsb = input_list[26]
        
        current_value = self._to_signed_24bit(msb, middle, lsb)
        
        if self.last_valid_value is None:
            self.last_valid_value = current_value
            return False
        
        difference = abs(current_value - self.last_valid_value)
        if difference > 5000:
            print(f'Corrupted data detected counter: {self.counter}')
            self.counter += 1
            return False
        else:
            self.last_valid_value = current_value
            return True
    
    def read_data(self):
        button_state = self.button_line.get_value()
        
        if button_state == 1:
            self.test_drdy = 10
        
        if self.test_drdy == 10 and button_state == 0:
            self.test_drdy = 0
            
            output = self.ads1.read_bytes(27)
            output_2 = self.ads2.read_bytes(27)
            
            if not self._is_valid_input(output_2):
                return None, None
            
            if output_2[0] == 192 and output_2[1] == 0 and output_2[2] == 8:
                result = [0] * 27
                result_2 = [0] * 27
                
                # Process first ADS1299
                for a in range(3, 25, 3):
                    voltage_1 = (output[a] << 8) | output[a + 1]
                    voltage_1 = (voltage_1 << 8) | output[a + 2]
                    convert_voltage = voltage_1 | self.data_test
                    
                    if convert_voltage == self.data_check:
                        voltage_1_after_convert = voltage_1 - 16777214
                    else:
                        voltage_1_after_convert = voltage_1
                    
                    channel_num = a // 3
                    result[channel_num] = round(1000000 * 4.5 * (voltage_1_after_convert / 16777215), 2)
                
                # Process second ADS1299
                for a in range(3, 25, 3):
                    voltage_1 = (output_2[a] << 8) | output_2[a + 1]
                    voltage_1 = (voltage_1 << 8) | output_2[a + 2]
                    convert_voltage = voltage_1 | self.data_test
                    
                    if convert_voltage == self.data_check:
                        voltage_1_after_convert = voltage_1 - 16777214
                    else:
                        voltage_1_after_convert = voltage_1
                    
                    channel_num = a // 3
                    result_2[channel_num] = round(1000000 * 4.5 * (voltage_1_after_convert / 16777215), 2)
                
                return result[1:9], result_2[1:9]  # Return 8 channels from each ADS1299
        
        return None, None
    
    def start_acquisition(self):
        self.is_running = True
    
    def stop_acquisition(self):
        self.is_running = False

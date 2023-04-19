r"""
    Connect Xsens dot sensors. Modified from https://github.com/adamkewley/xdc
"""


__all__ = ['Acceleration', 'AngularVelocity', 'BatteryCharacteristic',
           'ClipCountAcc', 'ClipCountGyr', 'ControlCharacteristic', 'DeviceControlCharacteristic',
           'DeviceInfoCharacteristic', 'DeviceReportCharacteristic', 'Dot', 'Dq', 'Dv', 'EulerAngles',
           'FreeAcceleration', 'LongPayloadCharacteristic', 'LongPayloadCustomMode4', 'MagneticField',
           'MediumPayloadCharacteristic', 'MediumPayloadCompleteEuler', 'MediumPayloadCompleteQuaternion',
           'MediumPayloadCustomMode1', 'MediumPayloadCustomMode2', 'MediumPayloadCustomMode3',
           'MediumPayloadDeltaQuantities', 'MediumPayloadDeltaQuantitiesWithMag', 'MediumPayloadExtendedEuler',
           'MediumPayloadExtendedQuaternion', 'MediumPayloadHighFidelity', 'MediumPayloadHighFidelityWithMag',
           'MediumPayloadRateQuantities', 'MediumPayloadRateQuantitiesWithMag', 'OrientationResetControlCharacteristic',
           'OrientationResetStatusCharacteristic', 'Quaternion', 'ShortPayloadCharacteristic',
           'ShortPayloadFreeAcceleration', 'ShortPayloadOrientationEuler', 'ShortPayloadOrientationQuaternion',
           'Status', 'Timestamp', 'adevice_control_read', 'adevice_control_write', 'adevice_info_read',
           'adisable_power_on_by_usb_plug_in', 'aenable_power_on_by_usb_plug_in', 'afind_by_address',
           'afind_dot_by_address', 'aidentify', 'ais_dot', 'apower_off', 'areset_output_rate', 'ascan', 'ascan_all',
           'aset_filter_profile_index', 'aset_output_rate', 'device_control_read', 'device_control_write',
           'device_info_read', 'disable_power_on_by_usb_plug_in', 'enable_power_on_by_usb_plug_in', 'find_by_address',
           'find_dot_by_address', 'identify', 'is_dot', 'power_off', 'reset_output_rate', 'scan', 'scan_all',
           'set_filter_profile_index', 'set_output_rate']


import struct  # for unpacking floating-point data
import asyncio
from typing import Any  # for async BLE IO
from bleak import BleakScanner, BleakClient  # for BLE communication


# returns a pretty-printed representation of an arbitrary class
def _pretty_print(obj: Any) -> str:
    return f"{obj.__class__.__name__}({', '.join('%s=%s' % item for item in vars(obj).items())})"


# a helper class that encapsulates a reader "cursor" that indexes a
# location in an array of bytes. Provides methods for reading multi-byte
# sequences (floats, ints, etc.) from the array, advancing the cursor as
# it reads. Multi-byte sequences are parsed according to XSens's binary
# spec (namely, little-endian with IEE754 floats)
class _ResponseReader:

    # initializes a reader that's pointing to the start of `data`
    def __init__(self, data):
        self.pos = 0
        self.data = data

    # returns number of remaining bytes that the reader can still read
    def remaining(self) -> int:
        return len(self.data) - self.pos

    # read `n` raw bytes from the reader's current position and advance
    # the current position by `n`
    def read_bytes(self, n: int) -> bytes:
        rv = self.data[self.pos:self.pos+n]
        self.pos += n
        return rv

    # read 1 byte as an unsigned int
    def read_u8(self) -> int:
        return int.from_bytes(self.read_bytes(1), "little", signed=False)

    # read 2 bytes as an unsigned little-endian int
    def read_u16(self) -> int:
        return int.from_bytes(self.read_bytes(2), "little", signed=False)

    # read 4 bytes as an unsigned little-endian int
    def read_u32(self) -> int:
        return int.from_bytes(self.read_bytes(4), "little", signed=False)

    # read 8 bytes as an unsigned little-endian int
    def read_u64(self) -> int:
        return int.from_bytes(self.read_bytes(8), "little", signed=False)

    # read 4 bytes as a IEE754 floating point number
    def read_f32(self) -> float:
        return struct.unpack('f', self.read_bytes(4))[0]


# SERVICES (as defined in the BLE spec)
#
# the XSens DOT provides BLE services with the following prefixes on the
# UUID:
#
# 0x1000 (e.g. 15171000-4947-...): configuration service
# 0x2000                         : measurement service
# 0x3000                         : charging status/battery level service
# 0x7000                         : message service

# Configuration Service: Device Info Characteristic (sec 2.1, p8 in the BLE spec)
#
# read-only characteristic for top-level device information
class DeviceInfoCharacteristic:
    UUID = "15171001-4947-11E9-8646-D663BD873D93"
    SIZE = 34

    # returns a `DeviceInfoCharacteristic` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= DeviceInfoCharacteristic.SIZE

        rv = DeviceInfoCharacteristic()
        rv.address = reader.read_bytes(6)
        rv.version_major = reader.read_u8()
        rv.version_minor = reader.read_u8()
        rv.version_revision = reader.read_u8()
        rv.build_year = reader.read_u16()  # 2019 ~ 2100
        rv.build_month = reader.read_u8()  # 1 ~ 12
        rv.build_date = reader.read_u8()  # 1 ~ 31
        rv.build_hour = reader.read_u8()  # 1 ~ 23
        rv.build_minute = reader.read_u8()  # 0 ~ 59
        rv.build_second = reader.read_u8()  # 0 ~ 59
        rv.softdevice_version = reader.read_u32()
        rv.serial_number = reader.read_u64()
        rv.short_product_code = reader.read_bytes(6)  # e.g. "XS-T01"

        return rv

    # returns a `DeviceInfoCharacteristic` parsed from bytes
    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return DeviceInfoCharacteristic.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# Configuration Service: Device Control Characteristic (sec 2.2, p9 in the BLE spec)
#
# read/write characteristic for top-level control (i.e. mode) of the DOT.
class DeviceControlCharacteristic:
    UUID = "15171002-4947-11E9-8646-D663BD873D93"
    SIZE = 16

    # returns a `DeviceControlCharacteristic` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= DeviceControlCharacteristic.SIZE

        rv = DeviceControlCharacteristic()
        rv.visit_index = reader.read_u8()
        rv.identifying = reader.read_u8()
        rv.power_options = reader.read_u8()
        rv.power_saving_timeout_x_mins = reader.read_u8()  # 0 ~ 30
        rv.power_saving_timeout_x_secs = reader.read_u8()  # 0 ~ 60
        rv.power_saving_timeout_y_mins = reader.read_u8()  # 0 ~ 30
        rv.power_saving_timeout_y_secs = reader.read_u8()  # 0 ~ 60
        rv.device_tag_len = reader.read_u8()
        rv.device_tag = reader.read_bytes(16)  # default "Xsens DOT"
        rv.output_rate = reader.read_u16()  # 1, 4, 10, 12, 15, 20, 30, 60, 120hz
        rv.filter_profile_index = reader.read_u8()
        rv.reserved = reader.read_bytes(5)  # just in case someone's interested

        return rv

    # returns a `DeviceControlCharacteristic` parsed from bytes
    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return DeviceControlCharacteristic.from_reader(reader)

    # returns bytes serialized from the `DeviceControlCharacteristic`'s data
    def to_bytes(self):
        rv = bytearray()
        rv += self.visit_index.to_bytes(1, "little")
        rv += self.identifying.to_bytes(1, "little")
        rv += self.power_options.to_bytes(1, "little")
        rv += self.power_saving_timeout_x_mins.to_bytes(1, "little")
        rv += self.power_saving_timeout_x_secs.to_bytes(1, "little")
        rv += self.power_saving_timeout_y_mins.to_bytes(1, "little")
        rv += self.power_saving_timeout_y_secs.to_bytes(1, "little")
        rv += self.device_tag_len.to_bytes(1, "little")
        rv += self.device_tag
        rv += self.output_rate.to_bytes(2, "little")
        rv += self.filter_profile_index.to_bytes(1, "little")
        rv += self.reserved

        return rv

    def __repr__(self):
        return _pretty_print(self)


# Configuration Service: Device Report Characteristic (sec 2.3, p10 in the BLE spec)
#
# notification characteristic for various events the DOT may emit (e.g.
# power off, button pressed)
class DeviceReportCharacteristic:
    UUID = "15171004-4947-11E9-8646-D663BD873D93"
    SIZE = 36

    # returns a `DeviceReportCharacteristic` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= DeviceReportCharacteristic.SIZE

        end = reader.pos + DeviceReportCharacteristic.SIZE

        rv = DeviceReportCharacteristic()
        rv.typeid = reader.read_u8()

        if rv.typeid == 1:
            # power off report
            pass
        elif rv.typeid == 4:
            # power saving report
            pass
        elif rv.typeid == 5:
            # button callback report
            rv.length = reader.read_u8()
            if rv.length == 4:
                rv.timestamp = reader.read_u32()
            elif rv.length == 8:
                rv.timestamp = reader.read_u64()
        else:
            # unknown report type
            pass

        # record unused bytes in-case future DOT devices make them
        # contain useful information (so users can still use this lib
        # to parse them afterwards)
        rv.unused = reader.read_bytes(end - reader.pos)

        return rv

    # returns a `DeviceReportCharacteristic` parsed from bytes
    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return DeviceReportCharacteristic.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# Measurement Service: Control Characteristic (sec 3.1, p12 in the BLE spec)
#
# read/write characteristic that controls the DOT's measurement state (i.e. if/what
# measurements are enabled)
class ControlCharacteristic:
    UUID = "15172001-4947-11E9-8646-D663BD873D93"
    SIZE = 3

    # returns a `ControlCharacteristic` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= ControlCharacteristic.SIZE

        rv = ControlCharacteristic()
        rv.Type = reader.read_u8()
        rv.action = reader.read_u8()
        rv.payload_mode = reader.read_u8()

        return rv

    # parse bytes as characteristic data
    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return ControlCharacteristic.from_reader(reader)

    # convert characteristic data back to bytes
    def to_bytes(self):
        assert self.Type < 0xff
        assert self.action <= 1
        assert self.payload_mode <= 24

        rv = bytearray()
        rv += self.Type.to_bytes(1, "little")
        rv += self.action.to_bytes(1, "little")
        rv += self.payload_mode.to_bytes(1, "little")

        return rv

    def __repr__(self):
        return _pretty_print(self)


# the next bunch of classes are parsers etc. for the wide variety of measurement structs
# that the measurement service can emit

# measurement data (sec. 3.5 in BLE spec): timestamp: "Timestamp of the sensor in microseconds"
class Timestamp:
    SIZE = 4

    # returns a `Timestamp` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= Timestamp.SIZE

        rv = Timestamp()
        rv.microseconds = reader.read_u32()

        return rv

    def __repr__(self):
        return _pretty_print(self)


# measurement data (sec. 3.5 in BLE spec): quaternion: "The orientation expressed as a quaternion"
class Quaternion:
    SIZE = 16

    # returns a `Quaternion` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= Quaternion.SIZE

        rv = Quaternion()
        rv.w = reader.read_f32()
        rv.x = reader.read_f32()
        rv.y = reader.read_f32()
        rv.z = reader.read_f32()

        return rv

    def __repr__(self):
        return _pretty_print(self)


# measurement data (sec. 3.5 in BLE spec): euler angles: "The orientation expressed as Euler angles, degree"
class EulerAngles:
    SIZE = 12

    # returns a `EulerAngles` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= EulerAngles.SIZE

        rv = EulerAngles()
        rv.x = reader.read_f32()
        rv.y = reader.read_f32()
        rv.z = reader.read_f32()

        return rv

    def __repr__(self):
        return _pretty_print(self)


# measurement data (sec. 3.5 in the BLE spec): free acceleration: "Acceleration in local earth coordinate 
# and the local gravity is deducted, m/s^2"
class FreeAcceleration:
    SIZE = 12

    # returns a `FreeAcceleration` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= FreeAcceleration.SIZE

        rv = FreeAcceleration()
        rv.x = reader.read_f32()
        rv.y = reader.read_f32()
        rv.z = reader.read_f32()

        return rv

    def __repr__(self):
        return _pretty_print(self)


# measurement data (sec. 3.5 in BLE spec): dq: "Orientation change during a time interval"
class Dq:
    SIZE = 16

    # returns a `Dq` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= Dq.SIZE

        rv = Dq()
        rv.w = reader.read_f32()
        rv.x = reader.read_f32()
        rv.y = reader.read_f32()
        rv.z = reader.read_f32()

        return rv

    def __repr__(self):
        return _pretty_print(self)


# measurement data (sec. 3.5 in the BLE spec): dv: "Velocity change during a time interval, m/s"
class Dv:
    SIZE = 12

    # returns a `Dv` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= Dv.SIZE

        rv = Dv()
        rv.x = reader.read_f32()
        rv.y = reader.read_f32()
        rv.z = reader.read_f32()

        return rv

    def __repr__(self):
        return _pretty_print(self)


# measurement data (sec. 3.5 in the BLE spec): Acceleration: "Calibrated acceleration in sensor coordinate, m/s^2"
class Acceleration:
    SIZE = 12

    # returns a `Acceleration` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= Acceleration.SIZE

        rv = Acceleration()
        rv.x = reader.read_f32()
        rv.y = reader.read_f32()
        rv.z = reader.read_f32()

        return rv

    def __repr__(self):
        return _pretty_print(self)


# measurement data (sec. 3.5 in the BLE spec): Angular Velocity: "Rate of turn in sensor coordinate, dps"
class AngularVelocity:
    SIZE = 12

    # returns an `AngularVelocity` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= AngularVelocity.SIZE

        rv = AngularVelocity()
        rv.x = reader.read_f32()
        rv.y = reader.read_f32()
        rv.z = reader.read_f32()

        return rv

    def __repr__(self):
        return _pretty_print(self)


# measurement data (sec. 3.5 in the BLE spec): Magnetic Field: "Magnetic field in the sensor coordinate, a.u."
class MagneticField:
    SIZE = 6

    # returns a `MagneticField` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MagneticField.SIZE

        rv = MagneticField()
        rv.x = reader.read_bytes(2)
        rv.y = reader.read_bytes(2)
        rv.z = reader.read_bytes(2)

        return rv

    def __repr__(self):
        return _pretty_print(self)


# measurement data (sec. 3.5 in the BLE spec): Status: "See section 3.5.1 of the BLE spec"
class Status:
    SIZE = 2

    # returns a `Status` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= Status.SIZE

        rv = Status()
        rv.value = reader.read_u16()

        return rv

    def __repr__(self):
        return _pretty_print(self)


# measurement data (sec. 3.5. in the BLE spec): ClipCountAcc: "Count of ClipAcc in status"
class ClipCountAcc:
    SIZE = 1

    # returns a `ClipCountAcc` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= ClipCountAcc.SIZE

        rv = ClipCountAcc()
        rv.value = reader.read_u8()

        return rv

    def __repr__(self):
        return _pretty_print(self)


# measurement data (sec. 3.5. in the BLE spec): ClipCountGyr: "Count of ClipGyr in status"
class ClipCountGyr:
    SIZE = 1

    # returns a `ClipCountGyr` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= ClipCountGyr.SIZE

        rv = ClipCountGyr()
        rv.value = reader.read_u8()

        return rv

    def __repr__(self):
        return _pretty_print(self)


# data for long-payload measurements
#
# the DOT can emit data in long (63-byte), medium (40-byte), and short (20-byte) lengths.
# What those bytes parse out to depends on the payload mode (see "Measurement Service Control
# Characteristic") is set to.
class LongPayloadCharacteristic:
    UUID = "15172002-4947-11E9-8646-D663BD873D93"


# a long payload measurement response called "Custom Mode 4" in the BLE spec
class LongPayloadCustomMode4:
    SIZE = 51

    # no parser: it's not officially supported by XSens


# data for medium-payload measurements
#
# the DOT can emit data in long (63-byte), medium (40-byte), and short (20-byte) lengths.
# What those bytes parse out to depends on the payload mode (see "Measurement Service Control
# Characteristic") is set to.
class MediumPayloadCharacteristic:
    UUID = "15172003-4947-11E9-8646-D663BD873D93"


# a medium-payload measurement response that contains "Extended Quaternion" data
class MediumPayloadExtendedQuaternion:
    SIZE = 36

    # returns a `MediumPayloadExtendedQuaternion` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MediumPayloadExtendedQuaternion.SIZE

        rv = MediumPayloadExtendedQuaternion()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.quaternion = Quaternion.from_reader(reader)
        rv.free_acceleration = FreeAcceleration.from_reader(reader)
        rv.status = Status.from_reader(reader)
        rv.clip_count_acc = ClipCountAcc.from_reader(reader)
        rv.clip_count_gyr = ClipCountGyr.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return MediumPayloadExtendedQuaternion.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a medium-payload measurement response that contains "Complete Quaternion" data
class MediumPayloadCompleteQuaternion:
    SIZE = 32

    # returns a `MediumPayloadCompleteQuaternion` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MediumPayloadCompleteQuaternion.SIZE

        rv = MediumPayloadCompleteQuaternion()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.quaternion = Quaternion.from_reader(reader)
        rv.free_acceleration = FreeAcceleration.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return MediumPayloadCompleteQuaternion.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a medium-payload measurement response that contains "Extended Euler" data
class MediumPayloadExtendedEuler:
    SIZE = 32

    # returns a `MediumPayloadExtendedEuler` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MediumPayloadExtendedEuler.SIZE

        rv = MediumPayloadExtendedEuler()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.euler = EulerAngles.from_reader(reader)
        rv.free_acceleration = FreeAcceleration.from_reader(reader)
        rv.status = Status.from_reader(reader)
        rv.clip_count_acc = ClipCountAcc.from_reader(reader)
        rv.clip_count_gyr = ClipCountGyr.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return MediumPayloadExtendedEuler.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a medium-payload measurement response that contains "Complete Euler" data
class MediumPayloadCompleteEuler:
    SIZE = 28

    # returns a `MediumPayloadCompleteEuler` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MediumPayloadCompleteEuler.SIZE

        rv = MediumPayloadCompleteEuler()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.euler = EulerAngles.from_reader(reader)
        rv.free_acceleration = FreeAcceleration.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return MediumPayloadCompleteEuler.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a medium-payload measurement reponse that contains "High Fidelity (with mag)" data
class MediumPayloadHighFidelityWithMag:
    SIZE = 35

    # no parser: XSens claims you need to use the SDK to get this


# a medium-payload measurement response that contains "High Fidelity" data
class MediumPayloadHighFidelity:
    SIZE = 29

    # no parser: XSens claims you need to use the SDK to get this


# a medium-payload measurement response that contains "Delta Quantities (with Mag)" data
class MediumPayloadDeltaQuantitiesWithMag:
    SIZE = 38

    # returns a `MediumPayloadDeltaQuantitiesWithMag` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MediumPayloadDeltaQuantitiesWithMag.SIZE

        rv = MediumPayloadDeltaQuantitiesWithMag()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.dq = Dq.from_reader(reader)
        rv.dv = Dv.from_reader(reader)
        rv.magnetic_field = MagneticField.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return MediumPayloadDeltaQuantitiesWithMag.from_reader(reader)


# a medium-payload measurement response that contains "Delta Quantites" data
class MediumPayloadDeltaQuantities:
    SIZE = 32

    # returns a `MediumPayloadDeltaQuantities` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MediumPayloadDeltaQuantities.SIZE

        rv = MediumPayloadDeltaQuantities()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.dq = Dq.from_reader(reader)
        rv.dv = Dv.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return MediumPayloadDeltaQuantities.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a medium-payload measurement response that contains "Rate Quantities (with Mag)" data
class MediumPayloadRateQuantitiesWithMag:
    SIZE = 34

    # returns a `MediumPayloadRateQuantitiesWithMag` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MediumPayloadRateQuantitiesWithMag.SIZE

        rv = MediumPayloadRateQuantitiesWithMag()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.acceleration = Acceleration.from_reader(reader)
        rv.angular_velocity = AngularVelocity.from_reader(reader)
        rv.magnetic_field = MagneticField.from_reader(reader)

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return MediumPayloadRateQuantitiesWithMag.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a medium-payload measurement response that contains "Rate Quantities" data
class MediumPayloadRateQuantities:
    SIZE = 28

    # returns a `MediumPayloadRateQuantities` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MediumPayloadRateQuantities.SIZE

        rv = MediumPayloadRateQuantities()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.acceleration = Acceleration.from_reader(reader)
        rv.angular_velocity = AngularVelocity.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return MediumPayloadRateQuantities.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a medium-payload measurement response that contains "Custom Mode 1" data
class MediumPayloadCustomMode1:
    SIZE = 40

    # returns a `MediumPayloadCustomMode1` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MediumPayloadCustomMode1.SIZE

        rv = MediumPayloadCustomMode1()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.euler = EulerAngles.from_reader(reader)
        rv.free_acceleration = FreeAcceleration.from_reader(reader)
        rv.angular_velocity = AngularVelocity.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return MediumPayloadCustomMode1.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a medium-payload measurement response that contains "Custom Mode 2" data
class MediumPayloadCustomMode2:
    SIZE = 34

    # returns a `MediumPayloadCustomMode2` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MediumPayloadCustomMode2.SIZE

        rv = MediumPayloadCustomMode2()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.euler = EulerAngles.from_reader(reader)
        rv.free_acceleration = FreeAcceleration.from_reader(reader)
        rv.magnetic_field = MagneticField.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return MediumPayloadCustomMode2.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a medium-payload measurement response that contains "Custom Mode 3" data
class MediumPayloadCustomMode3:
    SIZE = 32

    # returns a `MediumPayloadCustomMode3` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= MediumPayloadCustomMode3.SIZE

        rv = MediumPayloadCustomMode3()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.quaternion = Quaternion.from_reader(reader)
        rv.angular_velocity = AngularVelocity.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return MediumPayloadCustomMode3.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# data for short-payload measurements
#
# the DOT can emit data in long (63-byte), medium (40-byte), and short (20-byte) lengths.
# What those bytes parse out to depends on the payload mode (see "Measurement Service Control
# Characteristic") is set to.
class ShortPayloadCharacteristic:
    UUID = "15172004-4947-11E9-8646-D663BD873D93"


# a short-payload measurement response that contains "Orientation Euler" data
class ShortPayloadOrientationEuler:
    SIZE = 16

    # returns a `ShortPayloadOrientationEuler` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= ShortPayloadOrientationEuler.SIZE

        rv = ShortPayloadOrientationEuler()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.euler = EulerAngles.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return ShortPayloadOrientationEuler.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a short-payload measurement response that contains "Orientation Quaternion" data
class ShortPayloadOrientationQuaternion:
    SIZE = 20

    # returns a `ShortPayloadOrientationQuaternion` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= ShortPayloadOrientationQuaternion.SIZE

        rv = ShortPayloadOrientationQuaternion()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.quaternion = Quaternion.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return ShortPayloadOrientationQuaternion.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a short-payload measurement response that contains "Free Acceleration" data
class ShortPayloadFreeAcceleration:
    SIZE = 16

    # returns a `ShortPayloadFreeAcceleration` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= ShortPayloadFreeAcceleration.SIZE

        rv = ShortPayloadFreeAcceleration()
        rv.timestamp = Timestamp.from_reader(reader)
        rv.free_acceleration = FreeAcceleration.from_reader(reader)

        return rv

    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return ShortPayloadFreeAcceleration.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# Measurement Service: Orientation Reset Control Characteristic (sec. 3.6, p17 in the BLE spec)
class OrientationResetControlCharacteristic:
    UUID = "15172006-4947-11E9-8646-D663BD873D93"
    SIZE = 2

    # returns a `OrientationResetControlCharacteristic` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= OrientationResetControlCharacteristic.SIZE

        rv = OrientationResetControlCharacteristic()
        rv.Type = reader.read_u16()

        return rv

    # returns a `OrientationResetControlCharacteristic` parsed from a byte sequence
    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return OrientationResetControlCharacteristic.from_reader(reader)

    # returns a serialized representation of the `OrientationResetControlCharacteristic`
    def to_bytes(self):
        rv = bytearray()
        rv += self.Type.to_bytes(2, "little")
        return rv

    def __repr__(self):
        return _pretty_print(self)


# Measurement Service: Orientation Reset Status Characteristic (sec. 3.7, p17 in the BLE spec)
class OrientationResetStatusCharacteristic:
    UUID = "15172007-4947-11E9-8646-D663BD873D93"
    SIZE = 1

    # returns a `OrientationResetStatusCharacteristic` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= OrientationResetStatusCharacteristic.SIZE

        rv = OrientationResetStatusCharacteristic()
        rv.reset_result = reader.read_u8()

        return rv

    # returns a `OrientationResetStatusCharacteristic` parsed from a byte sequence
    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return OrientationResetStatusCharacteristic.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# OrientationResetDataCharacteristic: not for public use


# Battery Service: Battery Characteristic (sec. 4.1, p19 in the BLE spec)
class BatteryCharacteristic:
    UUID = "15173001-4947-11E9-8646-D663BD873D93"
    SIZE = 2

    # returns a `BatteryCharacteristic` read from a `_ResponseReader`
    @staticmethod
    def from_reader(reader: _ResponseReader):
        assert reader.remaining() >= BatteryCharacteristic.SIZE

        rv = BatteryCharacteristic()
        rv.battery_level = reader.read_u8()
        rv.charging_status = reader.read_u8()

        return rv

    # returns a `BatteryCharacteristic` parsed from a byte sequence
    @staticmethod
    def from_bytes(bites):
        reader = _ResponseReader(bites)
        return BatteryCharacteristic.from_reader(reader)

    def __repr__(self):
        return _pretty_print(self)


# a class for connecting to, and using, an XSens DOT
#
# This class provides:
#
# - Methods for connecting to the underlying DOT device through the Bluetooth Low-Energy
#   (BLE) connection (`Dot.connect`, `Dot.disconnect`, `with`, and `async with`)
#
# - Low-level methods for reading, writing, and receiving notifications from the low-level
#   BLE characteristic the DOT exposes
#
# - High-level methods that use the low-level methods (e.g. turn the DOT on, identify it)
#
# This class acts as a lifetime wrapper around the underlying BLE connection, so
# you should use it in something like a `with` or `async with` block. Using the async
# API (methods prefixed with `a`) and `async with` block is better. The underlying BLE
# implementation is asynchronous. The synchronous API (other methods and plain `with`
# blocks) is more convenient, but has to hop into an asynchronous event loop until the
# entire method call is complete. The asynchronous equivalents have the opportunity to
# cooperatively yield so that other events can be processed while waiting for the response.
# It is practically a necessity to use the async API if handling a large amount of DOTs
# from one thread (otherwise, you will experience head-of-line blocking and have a harder
# time handling the side-effects of notications).
class Dot:

    # init/enter/exit: connect/disconnect to the DOT

    # init a new `Dot` instance
    #
    # initializes the underlying connection client, but does not connect to
    # the DOT. Use `.connect`/`.disconnect`, or (better) a context manager
    # (`with Dot(ble) as dot`), or (better again) an async context manager
    # (`async with Dot(ble) as dot`) to setup/teardown the connection
    def __init__(self, ble_device):
        self.client = BleakClient(ble_device)

    # automatically called when entering `async with` blocks
    async def __aenter__(self):
        await self.client.__aenter__()
        return self

    # automatically called when exiting `async with` blocks
    async def __aexit__(self, exc_type, value, traceback):
        await self.client.__aexit__(exc_type, value, traceback)

    # automatically called when entering (synchronous) `with` blocks
    def __enter__(self):
        asyncio.get_event_loop().run_until_complete(self.__aenter__())
        return self

    # automatically called when exiting (synchronous) `with` blocks
    def __exit__(self, exc_type, value, traceback):
        asyncio.get_event_loop().run_until_complete(self.__aexit__(exc_type, value, traceback))

    # (dis)connection methods

    # asynchronously establishes a connection to the DOT
    async def aconnect(self, timeout=10):
        return await self.client.connect(timeout=timeout)

    # synchronously establishes a connection to the DOT
    def connect(self, timeout=10):
        return asyncio.get_event_loop().run_until_complete(self.aconnect(timeout))

    # asynchronously terminates the connection to the DOT
    async def adisconnect(self):
        return await self.client.disconnect()

    # synchronously terminates the connection to the DOT
    def disconnect(self):
        return asyncio.get_event_loop().run_until_complete(self.adisconnect())

    # low-level characteristic accessors

    # asynchronously reads the "Device Info Characteristic" (sec. 2.1 in the BLE spec)
    async def adevice_info_read(self):
        resp = await self.client.read_gatt_char(DeviceInfoCharacteristic.UUID)
        return DeviceInfoCharacteristic.from_bytes(resp)

    # synchronously reads the "Device Info Characteristic" (sec. 2.1 in the BLE spec)
    def device_info_read(self):
        return asyncio.get_event_loop().run_until_complete(self.adevice_info_read())

    # asynchronously reads the "Device Control Characteristic" (sec. 2.2. in the BLE spec)
    async def adevice_control_read(self):
        resp = await self.client.read_gatt_char(DeviceControlCharacteristic.UUID)
        return DeviceControlCharacteristic.from_bytes(resp)

    # synchronously reads the "Device Control Characteristic" (sec. 2.2. in the BLE spec)
    def device_control_read(self):
        return asyncio.get_event_loop().run_until_complete(self.adevice_control_read())

    # asynchronously writes the "Device Control Characteristic" (sec. 2.2. in the BLE spec)
    #
    # arg must be a `DeviceControlCharacteristic` with its fields set to appropriate
    # values (read the BLE spec to see which values are supported)
    async def adevice_control_write(self, device_control_characteristic: DeviceControlCharacteristic):
        msg_bytes = device_control_characteristic.to_bytes()
        await self.client.write_gatt_char(DeviceControlCharacteristic.UUID, msg_bytes, True)

    # synchronously writes the "Device Control Characteristic" (sec. 2.2. in the BLE spec)
    #
    # arg must be a `DeviceControlCharacteristic` with its fields set to appropriate
    # values (read the BLE spec to see which values are supported)
    def device_control_write(self, device_control_characteristic: DeviceControlCharacteristic):
        asyncio.get_event_loop().run_until_complete(self.adevice_control_write(device_control_characteristic))

    # asynchronously enable notifications from the "Device Report Characteristic" (sec.
    # 2.3 in the BLE spec)
    #
    # once notifications are enabled, `callback` will be called with two arguments:
    # a message ID and the raw message bytes (which can be parsed using
    # `DeviceReportCharacteristic.parse`). Notifications arrive from the DOT whenever
    # a significant event happens (e.g. a button press). See the BLE spec for which
    # events trigger from which actions.
    async def adevice_report_start_notify(self, callback):
        await self.client.start_notify(DeviceReportCharacteristic.UUID, callback)

    # synchronously enable notifications from the "Device Report Characteristic" (sec.
    # 2.3 in the BLE spec)
    #
    # once notifications are enabled, `callback` will be called with two arguments:
    # a message ID and the raw message bytes (which can be parsed using
    # `DeviceReportCharacteristic.parse`). Notifications arrive from the DOT whenever
    # a significant event happens (e.g. a button press). See the BLE spec for which
    # events trigger from which actions.
    def device_report_start_notify(self, callback):
        asyncio.get_event_loop().run_until_complete(self.adevice_report_start_notify(callback))

    # asynchronously disable notifications from the "Device Report Characteristic" (sec.
    # 2.3 in the BLE spec)
    #
    # this disables notifications that were enabled by the `device_report_start_notify`
    # method. After this action completes, the `callback` in the enable call will no longer
    # be called
    async def adevice_report_stop_notify(self):
        await self.client.stop_notify(DeviceReportCharacteristic.UUID)

    # synchronously disable notifications from the "Device Report Characteristic" (sec.
    # 2.3 in the BLE spec)
    #
    # this disables notifications that were enabled by the `device_report_start_notify`
    # method. After this action completes, the `callback` in the enable call will no longer
    # be called
    def device_report_stop_notify(self):
        asyncio.get_event_loop().run_until_complete(self.adevice_report_stop_notify())

    # asynchronously read the "Control Characteristic" (sec. 3.1 in the BLE spec)
    async def acontrol_read(self):
        resp = await self.client.read_gatt_char(ControlCharacteristic.UUID)
        return ControlCharacteristic.from_bytes(resp)

    # synchronously read the "Control Characteristic" (sec. 3.1 in the BLE spec)
    def control_read(self):
        return asyncio.get_event_loop().run_until_complete(self.acontrol_read())

    # asynchronously write the "Control Characteristic" (sec. 3.1 in the BLE spec)
    async def acontrol_write(self, control_characteristic: ControlCharacteristic):
        msg_bytes = control_characteristic.to_bytes()
        await self.client.write_gatt_char(ControlCharacteristic.UUID, msg_bytes)

    # synchronously write the "Control Characteristic" (sec. 3.1 in the BLE spec)
    def control_write(self, control_characteristic: ControlCharacteristic):
        asyncio.get_event_loop().run_until_complete(self.acontrol_write(control_characteristic))

    # asynchronously read the "Orientation Reset Control Characteristic" (sec. 3.6 in the BLE spec)
    async def aorientation_reset_control_read(self):
        resp = await self.client.read_gatt_char(OrientationResetControlCharacteristic.UUID)
        return OrientationResetControlCharacteristic.from_bytes(resp)

    # synchronously read the "Orientation Reset Control Characteristic" (sec. 3.6 in the BLE spec)
    def orientation_reset_control_read(self):
        return asyncio.get_event_loop().run_until_complete(self.aorientation_reset_control_read())

    # asynchronously write the "Orientation Reset Control Characteristic" (sec. 3.6 in the BLE spec)
    async def aorientation_reset_control_write(self, orientation_reset_control_characteristic: OrientationResetControlCharacteristic):
        msg_bytes = orientation_reset_control_characteristic.to_bytes()
        await self.client.write_gatt_char(OrientationResetControlCharacteristic.UUID, msg_bytes)

    # synchronously write the "Orientation Reset Control Characteristic" (sec. 3.6 in the BLE spec)
    def orientation_reset_control_write(self, orientation_reset_control_characteristic: OrientationResetControlCharacteristic):
        asyncio.get_event_loop().run_until_complete(self.aorientation_reset_control_write(orientation_reset_control_characteristic))

    # asynchronously read the "Orientation Reset Status Characteristic" (sec. 3.7 in the BLE spec)
    async def aorientation_reset_status_read(self):
        resp = await self.client.read_gatt_char(OrientationResetStatusCharacteristic.UUID)
        return OrientationResetStatusCharacteristic.from_bytes(resp)

    # synchronously read the "Orientation Reset Status Characteristic" (sec. 3.7 in the BLE spec)
    def orientation_reset_status_read(self):
        return asyncio.get_event_loop().run_until_complete(self.aorientation_reset_status_read())

    # asynchronously subscribe to long-payload measurement notifications
    async def along_payload_start_notify(self, callback):
        await self.client.start_notify(LongPayloadCharacteristic.UUID, callback)

    # synchronously subscribe to long-payload measurement notifications
    def long_payload_start_notify(self, callback):
        asyncio.get_event_loop().run_until_complete(self.along_payload_start_notify(callback))

    async def along_payload_stop_notify(self):
        await self.client.stop_notify(LongPayloadCharacteristic.UUID)

    def long_payload_stop_notify(self):
        asyncio.get_event_loop().run_until_complete(self.along_payload_stop_notify())

    async def amedium_payload_start_notify(self, callback):
        await self.client.start_notify(MediumPayloadCharacteristic.UUID, callback)

    def medium_payload_start_notify(self, callback):
        asyncio.get_event_loop().run_until_complete(self.amedium_payload_start_notify(callback))

    async def amedium_payload_stop_notify(self):
        await self.client.stop_notify(MediumPayloadCharacteristic.UUID)

    def medium_payload_stop_notify(self):
        asyncio.get_event_loop().run_until_complete(self.amedium_payload_stop_notify())

    async def ashort_payload_start_notify(self, callback):
        await self.client.start_notify(ShortPayloadCharacteristic.UUID, callback)

    def short_payload_start_notify(self, callback):
        asyncio.get_event_loop().run_until_complete(self.ashort_payload_start_notify(callback))

    async def ashort_payload_stop_notify(self):
        await self.client.stop_notify(ShortPayloadCharacteristic.UUID)

    def short_payload_stop_notify(self):
        asyncio.get_event_loop().run_until_complete(self.ashort_payload_stop_notify())

    # asynchronously read the "Battery Characteristic" (sec. 4.1 in the BLE spec)
    async def abattery_read(self):
        resp = await self.client.read_gatt_char(BatteryCharacteristic.UUID)
        return BatteryCharacteristic.from_bytes(resp)

    # synchronously read the "Battery Characteristic" (sec. 4.1 in the BLE spec)
    def battery_read(self):
        return asyncio.get_event_loop().run_until_complete(self.abattery_read())

    # asynchronously enable battery notifications from the "Battery Characteristic" (see
    # sec. 4.1 in the BLE spec)
    async def abattery_start_notify(self, callback):
        await self.client.start_notify(BatteryCharacteristic.UUID, callback)

    # synchronously enable battery notifications from the "Battery Characteristic" (see
    # sec. 4.1 in the BLE spec)
    def battery_start_notify(self, callback):
        asyncio.get_event_loop().run_until_complete(self.abattery_start_notify(callback))

    # high-level operations

    # asynchronously requests that the DOT identifies itself
    #
    # (from BLE spec sec. 2.2.): The sensor LED will fast blink 8 times and then
    # a short pause in red, lasting for 10 seconds.
    async def aidentify(self):
        dc = await self.adevice_control_read()
        dc.visit_index = 0x01
        dc.identifying = 0x01
        await self.adevice_control_write(dc)

    # synchronously requests that the DOT identifies itself
    #
    # (from BLE spec sec. 2.2.): The sensor LED will fast blink 8 times and then
    # a short pause in red, lasting for 10 seconds.
    def identify(self):
        asyncio.get_event_loop().run_until_complete(self.aidentify())

    # asynchronously requests that the DOT powers itself off
    async def apower_off(self):
        dc = await self.adevice_control_read()
        dc.visit_index = 0x02
        dc.power_options = dc.power_options | 0x01
        await self.adevice_control_write(dc)

    # synchronously requests that the DOT powers itself off
    def power_off(self):
        asyncio.get_event_loop().run_until_complete(self.apower_off())

    # asynchronosly requests that the DOT should power itself on when plugged into
    # a USB (e.g. when charging)
    async def aenable_power_on_by_usb_plug_in(self):
        dc = await self.adevice_control_read()
        dc.visit_index = 0x02
        dc.poweroff = dc.poweroff | 0x02
        await self.adevice_control_write(dc)

    # synchronosly requests that the DOT should power itself on when plugged into
    # a USB (e.g. when charging)
    def enable_power_on_by_usb_plug_in(self):
        asyncio.get_event_loop().run_until_complete(self.aenable_power_on_by_usb_plug_in())

    # asynchronously requests that the DOT shouldn't power itself on when plugged into
    # a USB (e.g. when charging)
    async def adisable_power_on_by_usb_plug_in(self):
        dc = await self.adevice_control_read()
        dc.visit_index = 0x02
        dc.poweroff = dc.poweroff & ~0x02
        await self.adevice_control_write(dc)

    # synchronously requests that the DOT shouldn't power itself on when plugged into
    # a USB (e.g. when charging)
    def disable_power_on_by_usb_plug_in(self):
        asyncio.get_event_loop().run_until_complete(self.adisable_power_on_by_usb_plug_in())

    # asynchronously sets the output rate of the DOT
    #
    # (BLE spec sec. 2.2): only values 1,4,10,12,15,20,30,60,120hz are permitted
    async def aset_output_rate(self, rate):
        assert rate in {1, 4, 10, 12, 15, 20, 30, 60, 120}

        dc = await self.adevice_control_read()
        dc.visit_index = 0x10
        dc.output_rate = rate
        await self.adevice_control_write(dc)

    # synchronously sets the output rate of the DOT
    #
    # (BLE spec sec. 2.2): only values 1,4,10,12,15,20,30,60,120hz are permitted
    def set_output_rate(self, rate):
        asyncio.get_event_loop().run_until_complete(self.aset_output_rate(rate))

    # asynchronously resets the output rate of the DOT to its default value (60 hz)
    async def areset_output_rate(self):
        await self.aset_output_rate(60)  # default, according to BLE spec

    # synchronously resets the output rate of the DOT to its default value (60 hz)
    def reset_output_rate(self):
        asyncio.get_event_loop().run_until_complete(self.areset_output_rate())

    # asynchronously sets the "Filter Profile Index" field in the Device Control Characteristic
    #
    # this sets how the DOT filters measurements? No idea. See sec. 2.2 in the BLE
    # spec for a slightly better explanation
    async def aset_filter_profile_index(self, idx):
        assert idx in {0, 1}

        dc = await self.adevice_control_read()
        dc.visit_index = 0x20
        dc.filter_profile_index = idx
        await self.adevice_control_write(dc)

    # synchronously sets the "Filter Profile Index" field in the Device Control Characteristic
    #
    # this sets how the DOT filters measurements? No idea. See sec. 2.2 in the BLE
    # spec for a slightly better explanation
    def set_filter_profile_index(self, idx):
        asyncio.get_event_loop().run_until_complete(self.aset_filter_profile_index(idx))

    # asynchronously sets the "Filter Profile Index" of the DOT to "General"
    #
    # (from BLE spec sec. 2.2., table 8): "General" is the "Default for general human
    # motions"
    async def aset_filter_profile_to_general(self):
        await self.aset_filter_profile_index(0)

    # synchronously sets the "Filter Profile Index" of the DOT to "General"
    #
    # (from BLE spec sec. 2.2., table 8): "General" is the "Default for general human
    # motions"
    def set_filter_profile_to_general(self):
        asyncio.get_event_loop().run_until_complete(self.aset_filter_profile_to_general())

    # asynchronously sets the "Filter Profile Index" of the DOT to "Dynamic"
    #
    # (from BLE spec. sec. 2.2., table 8): "Dynamic" is "For fast and jerky human motions
    # like sprinting"
    async def aset_filter_profile_to_dynamic(self):
        await self.aset_filter_profile_index(1)

    # synchronously sets the "Filter Profile Index" of the DOT to "Dynamic"
    #
    # (from BLE spec. sec. 2.2., table 8): "Dynamic" is "For fast and jerky human motions
    # like sprinting"
    def set_filter_profile_to_dynamic(self):
        asyncio.get_event_loop().run_until_complete(self.aset_filter_profile_to_dynamic())

    # asynchronously start the measurement (BLE spec sec. 3.1)
    # enable BLE notification to get the measurement data for real-time streaming before calling this
    async def astart_streaming(self, payload_mode=3):
        while await self.ais_streaming():
            await self.astop_streaming()
            await asyncio.sleep(3)
        c = await self.acontrol_read()
        c.action = 1
        c.payload_mode = payload_mode
        await self.acontrol_write(c)

    # synchronously start the measurement (BLE spec sec. 3.1)
    # enable BLE notification to get the measurement data for real-time streaming before calling this
    def start_streaming(self, payload_mode=3):
        asyncio.get_event_loop().run_until_complete(self.astart_streaming(payload_mode))

    # asynchronously stop the measurement (BLE spec sec. 3.1)
    async def astop_streaming(self):
        c = await self.acontrol_read()
        c.action = 0
        await self.acontrol_write(c)

    # synchronously stop the measurement (BLE spec sec. 3.1)
    def stop_streaming(self):
        asyncio.get_event_loop().run_until_complete(self.astop_streaming())

    # asynchronously check if the sensor is streaming the measurement (BLE spec sec. 3.1)
    async def ais_streaming(self):
        c = await self.acontrol_read()
        return c.action == 1

    # synchronously check if the sensor is streaming the measurement (BLE spec sec. 3.1)
    def is_streaming(self):
        return asyncio.get_event_loop().run_until_complete(self.ais_streaming())

    # asynchronously reset heading (BLE spec sec. 3.6)
    # start real-time streaming before calling this
    async def areset_heading(self):
        assert await self.ais_streaming(), 'The heading reset must be executed during the measurement'
        while await self.ais_heading_reset():
            await self.arevert_heading_to_default()
            await asyncio.sleep(0.5)
        c = await self.aorientation_reset_control_read()
        c.Type = 1
        await self.aorientation_reset_control_write(c)
        ack = await self.aorientation_reset_status_read()
        if ack.reset_result != 1:
            print('Reset heading failed. Please try again.')

    # synchronously reset heading (BLE spec sec. 3.6)
    # start real-time streaming before calling this
    def reset_heading(self):
        asyncio.get_event_loop().run_until_complete(self.areset_heading())

    # asynchronously revert heading to default (BLE spec sec. 3.6)
    # start real-time streaming before calling this
    async def arevert_heading_to_default(self):
        assert await self.ais_streaming(), 'The heading revert must be executed during the measurement'
        c = await self.aorientation_reset_control_read()
        c.Type = 7
        await self.aorientation_reset_control_write(c)

    # synchronously revert heading to default (BLE spec sec. 3.6)
    # start real-time streaming before calling this
    def revert_heading_to_default(self):
        asyncio.get_event_loop().run_until_complete(self.arevert_heading_to_default())

    # asynchronously check if heading has been reset (BLE spec sec. 3.6)
    async def ais_heading_reset(self):
        assert await self.ais_streaming(), 'The heading reset must be executed during the measurement'
        c = await self.aorientation_reset_control_read()
        return c.Type == 1

    # synchronously check if heading has been reset (BLE spec sec. 3.6)
    def is_heading_reset(self):
        asyncio.get_event_loop().run_until_complete(self.ais_heading_reset())


# asynchronously returns `True` if the provided `bleak.backends.device.BLEDevice`
# is believed to be an XSens DOT sensor
async def ais_dot(bledevice):
    if bledevice.name and "Xsens DOT" in bledevice.name:
        # spec: 1.2: Scanning and Filtering: tag name is "Xsens DOT"
        #
        # other devices in the wild *may* have a name collision with this, but it's
        # unlikely
        return True
    elif bledevice.metadata.get("manufacturer_data") and bledevice.metadata["manufacturer_data"].get(2182):
        # spec: 1.2: Scanning and Filtering: Bluetooth SIG identifier for XSens Technologies B.V. is 2182 (0x0886)
        #
        # this ambiguously identifies an XSens DOT. XSens might recycle its bluetooth
        # ID for other products, though, so we need to check that the device actually
        # responds to a DOT-like packet
        try:
            async with Dot(bledevice) as dot:
                await dot.adevice_info_read()  # typical request
                return True
        except asyncio.exceptions.TimeoutError:
            return False


# synchronously returns `True` if the provided `bleak.backends.device.BLEDevice` is an
# XSens DOT
def is_dot(bledevice):
    return asyncio.get_event_loop().run_until_complete(ais_dot(bledevice))


# asynchronously returns a list of all (not just DOT) BLE devices that
# the host's bluetooth adaptor can see. Each element in the list is an
# instance of `bleak.backends.device.BLEDevice`
async def ascan_all(timeout=5):
    return await BleakScanner.discover(timeout=timeout)


# synchronously returns a list of all (not just DOT) BLE devices that the
# host's bluetooth adaptor can see. Each element in the list is an instance
# of `bleak.backends.device.BLEDevice`
def scan_all(timeout=5):
    return asyncio.get_event_loop().run_until_complete(ascan_all(timeout=timeout))


# asynchronously returns a list of all XSens DOTs that the host's bluetooth
# adaptor can see. Each element in the list is an instance of
# `bleak.backends.device.BLEDevice`
async def ascan(timeout=5):
    return [d for d in await ascan_all(timeout=timeout) if await ais_dot(d)]


# synchronously returns a list of all XSens DOTs that the host's bluetooth
# adaptor can see. Each element in the list is an instance of
# `bleak.backends.device.BLEDevice`
def scan(timeout=5):
    return asyncio.get_event_loop().run_until_complete(ascan(timeout=timeout))


# asynchronously returns a BLE device with the given identifier/address
#
# returns `None` if the device cannot be found (e.g. no connection, wrong
# address)
async def afind_by_address(device_identifier, timeout=5):
    return await BleakScanner.find_device_by_address(device_identifier, timeout=timeout)


# synchronously returns a BLE device with the given identifier/address
#
# returns `None` if the device cannot be found (e.g. no connection, wrong
# address)
def find_by_address(device_identifier, timeout=5):
    return asyncio.get_event_loop().run_until_complete(afind_by_address(device_identifier, timeout=timeout))


# asynchronously returns a BLE device with the given identifier/address if the
# device appears to be an XSens DOT
#
# effectively, the same as `afind_by_address` but with the extra stipulation that
# the given device must be a DOT
async def afind_dot_by_address(device_identifier, timeout=5):
    dev = await afind_by_address(device_identifier, timeout=timeout)

    if dev is None:
        return None  # device cannot be found
    elif not await ais_dot(dev):
        return None  # device exists but is not a DOT
    else:
        return dev


# synchronously returns a BLE device with the given identifier/address, if the device
# appears to be an XSens DOT
#
# effectively, the same as `find_by_address`, but with the extra stipulation that
# the given device must be a DOT
def find_dot_by_address(device_identifier, timeout=5):
    return asyncio.get_event_loop().run_until_complete(afind_dot_by_address(device_identifier, timeout=timeout))


# low-level characteristic accessors (free functions)


# asynchronously returns the "Device Info Characteristic" for the given DOT device
#
# see: sec 2.1 Device Info Characteristic in DOT BLE spec
async def adevice_info_read(bledevice):
    async with Dot(bledevice) as dot:
        return await dot.adevice_info_read()


# synchronously returns the "Device Info Characteristic" for the given DOT device
#
# see: sec 2.1: Device Info Characteristic in DOT BLE spec
def device_info_read(bledevice):
    return asyncio.get_event_loop().run_until_complete(adevice_info_read(bledevice))


# asynchronously returns the "Device Control Characteristic" for the given DOT device
#
# see: sec 2.2: Device Control Characteristic in DOT BLE spec
async def adevice_control_read(bledevice):
    async with Dot(bledevice) as dot:
        return await dot.adevice_control_read()


# synchronously returns the "Device Control Characteristic" for the given DOT device
#
# see: sec 2.2: Device Control Characteristic in DOT BLE spec
def device_control_read(bledevice):
    return asyncio.get_event_loop().run_until_complete(adevice_control_read(bledevice))


# asynchronously write the provided DeviceControlCharacteristic to the provided
# DOT device
async def adevice_control_write(bledevice, device_control_characteristic):
    async with Dot(bledevice) as dot:
        await dot.adevice_control_write(device_control_characteristic)


def device_control_write(bledevice, device_control_characteristic):
    asyncio.get_event_loop().run_until_complete(adevice_control_write(bledevice, device_control_characteristic))


# high-level operations (free functions)

async def aidentify(bledevice):
    async with Dot(bledevice) as dot:
        await dot.aidentify()


def identify(bledevice):
    asyncio.get_event_loop().run_until_complete(aidentify(bledevice))


async def apower_off(bledevice):
    async with Dot(bledevice) as dot:
        await dot.apower_off()


def power_off(bledevice):
    asyncio.get_event_loop().run_until_complete(apower_off(bledevice))


async def aenable_power_on_by_usb_plug_in(bledevice):
    async with Dot(bledevice) as dot:
        await dot.aenable_power_on_by_usb_plug_in()


def enable_power_on_by_usb_plug_in(bledevice):
    asyncio.get_event_loop().run_until_complete(aenable_power_on_by_usb_plug_in(bledevice))


async def adisable_power_on_by_usb_plug_in(bledevice):
    async with Dot(bledevice) as dot:
        await dot.adisable_power_on_by_usb_plug_in()


def disable_power_on_by_usb_plug_in(bledevice):
    asyncio.get_event_loop().run_until_complete(adisable_power_on_by_usb_plug_in(bledevice))


async def aset_output_rate(bledevice, rate):
    async with Dot(bledevice) as dot:
        await dot.aset_output_rate(rate)


def set_output_rate(bledevice, rate):
    asyncio.get_event_loop().run_until_complete(aset_output_rate(bledevice, rate))


async def areset_output_rate(bledevice):
    async with Dot(bledevice) as dot:
        await dot.areset_output_rate()


def reset_output_rate(bledevice):
    asyncio.get_event_loop().run_until_complete(areset_output_rate(bledevice))


async def aset_filter_profile_index(bledevice, idx):
    async with Dot(bledevice) as dot:
        await dot.aset_filter_profile_index(idx)


def set_filter_profile_index(bledevice, idx):
    asyncio.get_event_loop().run_until_complete(aset_filter_profile_index(bledevice, idx))


# example
if __name__ == '__main__':
    def on_device_report(message_id, message_bytes, sensor_id=-1):
        parsed = DeviceReportCharacteristic.from_bytes(message_bytes)
        print(sensor_id, parsed)

    def on_medium_payload_report(message_id, message_bytes, sensor_id=-1):
        parsed = MediumPayloadCompleteQuaternion.from_bytes(message_bytes)
        print(sensor_id, parsed)

    async def scan():
        all_devices = await ascan_all()
        print(all_devices)

    async def single_sensor():
        dot = await afind_by_address('D4:CA:6E:F1:99:A6')

        async with Dot(dot) as device:
            print(await device.adevice_info_read())
            print(await device.adevice_control_read())
            print(await device.abattery_read())

            await device.adevice_report_start_notify(on_device_report)
            await device.amedium_payload_start_notify(on_medium_payload_report)
            await device.astart_streaming(payload_mode=3)

            print('start')
            await asyncio.sleep(2)
            print('stop')

            await device.astop_streaming()
            await device.amedium_payload_stop_notify()
            await device.adevice_report_stop_notify()

    async def multiple_sensor():
        # Please use xsens dot app to synchronize the sensors first.
        from functools import partial

        devices = [
            'D4:CA:6E:F1:99:84',
            'D4:CA:6E:F1:99:A6',
            'D4:CA:6E:F1:9A:30',
            'D4:CA:6E:F1:99:82',
            'D4:22:CD:00:01:27',
            # 'D4:22:CD:00:03:0C',
            # 'D4:22:CD:00:00:8B',
            # 'D4:22:CD:00:06:5F',
            # 'D4:CA:6E:F0:C7:22',
            # 'D4:22:CD:00:02:87',
        ]

        print('finding devices ...')
        bledevices = await asyncio.gather(*[afind_by_address(d) for d in devices])
        dots = [Dot(d) for d in bledevices]

        print('connecting ...')
        await asyncio.gather(*[d.aconnect() for d in dots])

        print('reading battery infos ...', end=' ')
        battery_infos = await asyncio.gather(*[d.abattery_read() for d in dots])
        print(' '.join(['%d%%' % info.battery_level for info in battery_infos]))

        print('start the notifications ...', end=' ')
        await asyncio.gather(*[d.adevice_report_start_notify(partial(on_device_report, sensor_id=i)) for i, d in enumerate(dots)])
        await asyncio.gather(*[d.amedium_payload_start_notify(partial(on_medium_payload_report, sensor_id=i)) for i, d in enumerate(dots)])

        print('starting and reset heading ...')
        await asyncio.gather(*[d.astart_streaming(payload_mode=3) for d in dots])
        await asyncio.gather(*[d.areset_heading() for d in dots])
        await asyncio.sleep(2)

        print('stopping ...')
        await asyncio.gather(*[d.astop_streaming() for d in dots])
        await asyncio.gather(*[d.amedium_payload_stop_notify() for d in dots])
        await asyncio.gather(*[d.adevice_report_stop_notify() for d in dots])
        await asyncio.gather(*[d.adisconnect() for d in dots])

    def run_in_new_thread(coro):
        r"""
        Similar to `asyncio.run()`, but create a new thread.
        """
        def start_loop(_loop):
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        import threading
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=start_loop, args=(loop,))
        thread.setDaemon(True)
        thread.start()
        asyncio.run_coroutine_threadsafe(coro, loop)


    # asyncio.run(single_sensor())
    # asyncio.run(multiple_sensor())
    # run_in_new_thread(single_sensor())
    run_in_new_thread(multiple_sensor())

    import time
    while True:
        time.sleep(20)  # main thread blocking

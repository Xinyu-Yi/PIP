r"""
    Wrapper for an Xsens dot set.
"""


__all__ = ['XsensDotSet']


import asyncio
import time
from .xdc import *
from queue import Queue
import torch


_N = 10  # max 10 IMUs


class XsensDotSet:
    # _lock = [threading.Lock() for _ in range(_N)]   # lists are thread-safe
    _SZ = 180    # max queue size
    _loop = None
    _buffer = [Queue(180) for _ in range(_N)]
    _is_connected = False
    _is_started = False
    _pending_event = None

    @staticmethod
    def _on_device_report(message_id, message_bytes, sensor_id=-1):
        parsed = DeviceReportCharacteristic.from_bytes(message_bytes)
        print('IMU %d:' % sensor_id, parsed)

    @staticmethod
    def _on_medium_payload_report(message_id, message_bytes, sensor_id=-1):
        parsed = MediumPayloadCompleteQuaternion.from_bytes(message_bytes)
        q = XsensDotSet._buffer[sensor_id]
        if q.full():
            q.get()
        q.put(parsed)

    @staticmethod
    async def _multiple_sensor(devices: list):
        # Please use xsens dot app to synchronize the sensors first.
        # do not use asyncio.gather() and run these in parallel, it has bugs
        from functools import partial

        print('finding devices ...')
        dots = []
        for i, d in enumerate(devices):
            while True:
                try:
                    device = await afind_by_address(d, timeout=5)
                    if device is None:
                        print('\t[%d]' % i, 'device not detected')
                    else:
                        break
                except Exception as e:
                    print('\t[%d]' % i, e)
            dots.append(Dot(device))
            print('\t[%d]' % i, device)

        print('connecting ...')
        for i, d in enumerate(dots):
            while True:
                try:
                    await d.aconnect(timeout=8)
                    break
                except Exception as e:
                    print('\t[%d]' % i, e)
            print('\t[%d] connected' % i)

        print('reading battery infos ...')
        for i, d in enumerate(dots):
            info = await d.abattery_read()
            print('\t[%d] %d%%' % (i, info.battery_level))

        print('configuring the sensors ...')
        for i, d in enumerate(dots):
            await d.astop_streaming()
            await d.adevice_report_start_notify(partial(XsensDotSet._on_device_report, sensor_id=i))
            await d.amedium_payload_start_notify(partial(XsensDotSet._on_medium_payload_report, sensor_id=i))
            await d.aset_output_rate(60)

        print('sensors connected')
        XsensDotSet._is_connected = True

        while True:
            if XsensDotSet._pending_event == 0:   # close
                shutdown = False
                break
            elif XsensDotSet._pending_event == 1:   # shutdown
                shutdown = True
                break
            elif XsensDotSet._pending_event == 2:   # reset heading
                print('reset heading ...')
                for i, d in enumerate(dots):
                    await d.areset_heading()
                print('\theading is reset')
            elif XsensDotSet._pending_event == 3:   # revert heading
                print('revert heading ...')
                for i, d in enumerate(dots):
                    await d.arevert_heading_to_default()
                print('\theading is reverted to default')
            elif XsensDotSet._pending_event == 4:   # start streaming
                print('start streaming ...')
                for i, d in enumerate(dots):
                    await d.astart_streaming(payload_mode=3)
                XsensDotSet._is_started = True
                print('\tstreaming started')
            elif XsensDotSet._pending_event == 5:  # stop streaming
                print('stop streaming ...')
                for i, d in enumerate(dots):
                    await d.astop_streaming()
                XsensDotSet._is_started = False
                print('\tstreaming stopped')
            elif XsensDotSet._pending_event == 6:  # print battery
                print('reading battery infos ...')
                for i, d in enumerate(dots):
                    info = await d.abattery_read()
                    print('\t[%d] %d%%' % (i, info.battery_level))

            XsensDotSet._pending_event = None
            await asyncio.sleep(1)

        print('disconnecting ...')
        for i, d in enumerate(dots):
            await d.astop_streaming()
            await d.amedium_payload_stop_notify()
            await d.adevice_report_stop_notify()
            if shutdown:
                await d.apower_off()
                print('\t[%d] power off' % i)
            else:
                await d.adisconnect()
                print('\t[%d] disconnected' % i)

        XsensDotSet._is_started = False
        XsensDotSet._is_connected = False
        XsensDotSet._pending_event = None

    @staticmethod
    def _run_in_new_thread(coro):
        r"""
        Similar to `asyncio.run()`, but create a new thread.
        """
        def start_loop(_loop):
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        if XsensDotSet._loop is None:
            import threading
            XsensDotSet._loop = asyncio.new_event_loop()
            thread = threading.Thread(target=start_loop, args=(XsensDotSet._loop,))
            thread.setDaemon(True)
            thread.start()

        asyncio.run_coroutine_threadsafe(coro, XsensDotSet._loop)

    @staticmethod
    def _wait_for_pending_event():
        while XsensDotSet._pending_event is not None:
            time.sleep(0.3)

    @staticmethod
    def clear(i=-1):
        r"""
        Clear the cached measurements of the ith IMU. If i < 0, clear all IMUs.

        :param i: The index of the query sensor. If negative, clear all IMUs.
        """
        if i < 0:
            XsensDotSet._buffer = [Queue(XsensDotSet._SZ) for _ in range(_N)]  # max 10 IMUs
        else:
            XsensDotSet._buffer[i] = Queue(XsensDotSet._SZ)

    @staticmethod
    def is_started() -> bool:
        r"""
        Whether the sensors are started.
        """
        return XsensDotSet._is_started

    @staticmethod
    def is_connected() -> bool:
        r"""
        Whether the sensors are connected.
        """
        return XsensDotSet._is_connected

    @staticmethod
    def get(i: int, timeout=None, preserve_last=False):
        r"""
        Get the next measurements of the ith IMU. May be blocked.

        :param i: The index of the query sensor.
        :param timeout: If non-negative, block at most timeout seconds and raise an Empty error.
        :param preserve_last: If True, do not delete the measurement from the buffer if it is the last one.
        :return: timestamp (seconds), quaternion (wxyz), and free acceleration (m/s^2 in the global inertial frame)
        """
        if preserve_last and XsensDotSet._buffer[i].qsize() == 1:
            parsed = XsensDotSet._buffer[i].queue[0]
        else:
            parsed = XsensDotSet._buffer[i].get(block=True, timeout=timeout)
        t = parsed.timestamp.microseconds / 1e6
        q = torch.tensor([parsed.quaternion.w, parsed.quaternion.x, parsed.quaternion.y, parsed.quaternion.z])
        a = torch.tensor([parsed.free_acceleration.x, parsed.free_acceleration.y, parsed.free_acceleration.z])
        return t, q, a

    @staticmethod
    def async_connect(devices: list):
        r"""
        Connect to the sensors and start receiving the measurements.
        Only send the connecting command but will not be blocked.

        :param devices: List of Xsens dot addresses.
        """
        if not XsensDotSet.is_connected():
            print('Remember: use xsens dot app to synchronize the sensors first.')
            XsensDotSet._run_in_new_thread(XsensDotSet._multiple_sensor(devices))
        else:
            print('[Warning] connect failed: XsensDotSet is already connected.')

    @staticmethod
    def sync_connect(devices: list):
        r"""
        Connect to the sensors and start receiving the measurements. Block until finish.

        :param devices: List of Xsens dot addresses.
        """
        XsensDotSet.async_connect(devices)
        while not XsensDotSet.is_connected():
            time.sleep(1)

    @staticmethod
    def async_disconnect():
        r"""
        Stop reading and disconnect to the sensors.
        Only send the disconnecting command but will not be blocked.
        """
        if XsensDotSet.is_connected():
            XsensDotSet._pending_event = 0
        else:
            print('[Warning] disconnect failed: XsensDotSet is not connected.')

    @staticmethod
    def sync_disconnect():
        r"""
        Stop reading and disconnect to the sensors. Block until finish.
        """
        XsensDotSet.async_disconnect()
        while XsensDotSet.is_connected():
            time.sleep(1)

    @staticmethod
    def async_shutdown():
        r"""
        Stop reading and shutdown the sensors.
        Only send the shutdown command but will not be blocked.
        """
        if XsensDotSet.is_connected():
            XsensDotSet._pending_event = 1
        else:
            print('[Warning] shutdown failed: XsensDotSet is not connected.')

    @staticmethod
    def sync_shutdown():
        r"""
        Stop reading and shutdown the sensors. Block until finish.
        """
        XsensDotSet.async_shutdown()
        while XsensDotSet.is_connected():
            time.sleep(1)

    @staticmethod
    def reset_heading():
        r"""
        Reset sensor heading (yaw).
        """
        if XsensDotSet.is_started():
            XsensDotSet._pending_event = 2
            XsensDotSet._wait_for_pending_event()
        else:
            print('[Warning] reset heading failed: XsensDotSet is not started.')

    @staticmethod
    def revert_heading_to_default():
        r"""
        Revert sensor heading to default (yaw).
        """
        if XsensDotSet.is_started():
            XsensDotSet._pending_event = 3
            XsensDotSet._wait_for_pending_event()
        else:
            print('[Warning] revert heading failed: XsensDotSet is not started.')

    @staticmethod
    def start_streaming():
        r"""
        Start sensor streaming.
        """
        if not XsensDotSet.is_connected():
            print('[Warning] start streaming failed: XsensDotSet is not connected.')
        elif XsensDotSet.is_started():
            print('[Warning] start streaming failed: XsensDotSet is already started.')
        else:
            XsensDotSet._pending_event = 4
            XsensDotSet._wait_for_pending_event()

    @staticmethod
    def stop_streaming():
        r"""
        Stop sensor streaming.
        """
        if not XsensDotSet.is_connected():
            print('[Warning] stop streaming failed: XsensDotSet is not connected.')
        elif not XsensDotSet.is_started():
            print('[Warning] stop streaming failed: XsensDotSet is not started.')
        else:
            XsensDotSet._pending_event = 5
            XsensDotSet._wait_for_pending_event()

    @staticmethod
    def print_battery_info():
        r"""
        Print battery level infos.
        """
        if not XsensDotSet.is_connected():
            print('[Warning] print battery info failed: XsensDotSet is not connected.')
        else:
            XsensDotSet._pending_event = 6
            XsensDotSet._wait_for_pending_event()

    @staticmethod
    def set_buffer_len(n=180):
        r"""
        Set IMU buffer length. Cache the latest n measurements.

        :param n: Length of IMU buffer.
        """
        XsensDotSet._SZ = n
        XsensDotSet.clear()


# example
if __name__ == '__main__':
    # copy the following codes outside this package to run
    from articulate.utils.xsens import XsensDotSet
    from articulate.utils.bullet import RotationViewer
    from articulate.math import quaternion_to_rotation_matrix
    imus = [
        # 'D4:22:CD:00:36:80',
        # 'D4:22:CD:00:36:04',
        # 'D4:22:CD:00:32:3E',
        # 'D4:22:CD:00:35:4E',
        # 'D4:22:CD:00:36:03',
        # 'D4:22:CD:00:44:6E',
        # 'D4:22:CD:00:45:E6',
        'D4:22:CD:00:45:EC',
        'D4:22:CD:00:46:0F',
        'D4:22:CD:00:32:32',
    ]
    XsensDotSet.sync_connect(imus)
    XsensDotSet.start_streaming()
    XsensDotSet.reset_heading()
    with RotationViewer(len(imus)) as viewer:
        XsensDotSet.clear()
        for _ in range(60 * 10):  # 10s
            for i in range(len(imus)):
                t, q, a = XsensDotSet.get(i)
                viewer.update(quaternion_to_rotation_matrix(q).view(3, 3), i)
    XsensDotSet.sync_disconnect()

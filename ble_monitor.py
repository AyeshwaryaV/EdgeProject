"""BLE receiver for ESP32 OBD-II health telemetry."""

import asyncio
import re
from typing import Optional
from bleak import BleakClient, BleakScanner

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_NAME = "OBD_Health_Monitor"
NOTIFICATION_PATTERN = re.compile(r"H:(\d+),A:(\d),D:([0-9.+-eE]+)")


def parse_notification(notification: str) -> Optional[dict]:
    match = NOTIFICATION_PATTERN.search(notification)
    if not match:
        return None

    return {
        'health': int(match.group(1)),
        'anomaly': bool(int(match.group(2))),
        'distance': float(match.group(3)),
    }


async def notification_handler(sender, data):
    try:
        text = data.decode('utf-8').strip()
    except Exception:
        text = repr(data)

    parsed = parse_notification(text)
    if parsed:
        status = 'ANOMALY' if parsed['anomaly'] else 'NORMAL'
        print(f"[BLE] H:{parsed['health']}% | {status} | D:{parsed['distance']:.4f} | raw={text}")
    else:
        print(f"[BLE] Unknown payload: {text}")


async def main():
    print('[INFO] Scanning for BLE devices...')
    devices = await BleakScanner.discover(timeout=5.0)

    target = None
    for device in devices:
        if DEVICE_NAME in (device.name or ''):
            target = device
            break

    if target is None:
        print(f"[WARN] '{DEVICE_NAME}' not found. Listing nearby devices:")
        for device in devices:
            print(f"  - {device.name or '<unknown>'} [{device.address}]")
        return

    print(f"[INFO] Connecting to {target.name} [{target.address}]...")
    async with BleakClient(target.address) as client:
        if not await client.is_connected():
            print('[ERROR] Failed to connect')
            return

        print('[INFO] Connected. Subscribing to notifications...')
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        print('[INFO] Waiting for BLE notifications. Press Ctrl+C to stop.')
        try:
            while True:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            print('\n[INFO] Stopping BLE monitor...')
            await client.stop_notify(CHARACTERISTIC_UUID)


if __name__ == '__main__':
    asyncio.run(main())

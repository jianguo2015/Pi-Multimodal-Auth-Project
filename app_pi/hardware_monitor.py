import psutil

class PiHardwareMonitor:
    def __init__(self):
        self.is_pi = self._check_if_pi()

    def _check_if_pi(self):
        try:
            with open('/sys/firmware/devicetree/base/model', 'r') as m:
                if 'Raspberry Pi' in m.read(): return True
        except Exception: pass
        return False

    def get_cpu_temp(self):
        if not self.is_pi: return 0.0
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return float(f.read()) / 1000.0
        except Exception: return 0.0

    def get_system_stats(self):
        mem = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": mem.percent,
            "ram_used_mb": mem.used / (1024 * 1024),
            "temp_c": self.get_cpu_temp()
        }
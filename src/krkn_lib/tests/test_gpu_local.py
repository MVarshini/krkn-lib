import subprocess
import unittest
from unittest.mock import patch

from krkn_lib.gpu import Availability, snapshot_local_gpus


class LocalGpuSnapshotTests(unittest.TestCase):
    def test_local_snapshot_reports_values_and_missing_fields(self):
        xml = """<nvidia_smi_log><gpu><uuid>GPU-a</uuid>
            <fb_memory_usage><free>0 MiB</free><total>1024 MiB</total>
            </fb_memory_usage></gpu></nvidia_smi_log>"""

        def run(command, **kwargs):
            self.assertEqual(command, ("/host-driver/usr/bin/nvidia-smi", "-q", "-x"))
            self.assertEqual(kwargs["timeout"], 0.5)
            return subprocess.CompletedProcess(command, 0, xml, "")

        snapshot = snapshot_local_gpus(
            fields=["memory_free_bytes", "memory_used_bytes"],
            executable="/host-driver/usr/bin/nvidia-smi",
            timeout_seconds=0.5,
            run=run,
        )
        fields = {item.field: item for item in snapshot.measurements}
        self.assertEqual(fields["memory_free_bytes"].value, 0)
        self.assertEqual(fields["memory_free_bytes"].status, Availability.AVAILABLE)
        self.assertEqual(fields["memory_used_bytes"].status, Availability.UNKNOWN)
        self.assertIsNone(snapshot.context)

    def test_unavailable_driver_and_invalid_xml_do_not_supply_fake_measurements(self):
        with patch("krkn_lib.gpu.local.subprocess.run", side_effect=FileNotFoundError):
            missing = snapshot_local_gpus()
        with patch("krkn_lib.gpu.local.subprocess.run", return_value=subprocess.CompletedProcess(
            args=[], returncode=0, stdout="<incomplete", stderr=""
        )):
            malformed = snapshot_local_gpus()
        self.assertEqual(missing.errors, ["local nvidia-smi could not be started"])
        self.assertEqual(malformed.errors, ["local nvidia-smi returned invalid inventory XML"])
        self.assertEqual(missing.measurements + malformed.measurements, [])

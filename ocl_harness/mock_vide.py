import sys

import terra.devices.mock_camera as mockCamera
from terra.devices.base_camera import AutoExposureOptions
from terra.devices.continuous_reader import ContinuousCameraReader

for cls in (mockCamera.FilesystemCameraReader, ContinuousCameraReader):
    cls.set_auto_exposure_options = lambda *a, **k: None
    cls.set_auto_exposure = lambda *a, **k: None
    cls.set_auto_exposure_roi = lambda *a, **k: None
    cls.get_auto_exposure_options = lambda self: AutoExposureOptions()
    cls.get_auto_exposure = lambda self: False
    cls.get_auto_exposure_enabled = lambda self: False
    cls.get_exposure_range = lambda self: (1.0, 1000000.0)

sys.argv = ["vide", "--machine", "W3cj", "--config", "/var/lib/vide-gputest/vide.yaml",
            "--mock", "/tmp/realframes/w3y"]
from apps.run_vide import main

main()

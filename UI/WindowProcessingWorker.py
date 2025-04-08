from multiprocessing import Process, Value, Pipe
from multiprocessing.connection import Connection
import typing
import numpy as np
import cv2
import platform
import mss


def get_window_bounds(window_name):
    system = platform.system()
    
    if system == 'Windows':
        # Windows: use pygetwindow
        import pygetwindow as gw
        windows = gw.getWindowsWithTitle(window_name)
        if windows:
            window = windows[0]
            return {"top": window.top, "left": window.left, "width": window.width, "height": window.height}
    
    elif system == 'Darwin':  # macOS
        from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        for win in windows:
            if window_name in win.get('kCGWindowName', ''):
                bounds = win['kCGWindowBounds']
                return {"top": int(bounds['Y']), "left": int(bounds['X']), "width": int(bounds['Width']), "height": int(bounds['Height'])}
    
    elif system == 'Linux':
        import subprocess
        try:
            # Find the window ID and position with xwininfo and xdotool
            window_id = subprocess.check_output(["xdotool", "search", "--name", window_name]).decode().strip()
            if window_id:
                info = subprocess.check_output(["xwininfo", "-id", window_id]).decode()
                top = int([line for line in info.splitlines() if "Absolute upper-left Y" in line][0].split()[-1])
                left = int([line for line in info.splitlines() if "Absolute upper-left X" in line][0].split()[-1])
                width = int([line for line in info.splitlines() if "Width" in line][0].split()[-1])
                height = int([line for line in info.splitlines() if "Height" in line][0].split()[-1])
                return {"top": top, "left": left, "width": width, "height": height}
        except (subprocess.CalledProcessError, IndexError):
            print(f"Could not find window with title '{window_name}' using xdotool and xwininfo.")
    
    return None

def capture_window(window_name=None):
    monitor = get_window_bounds(window_name) if window_name else None
    
    if monitor:
        with mss.mss() as sct:
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
    else:
        print("No monitor set for capture.")
        return None


class PipeWriter:
    def __init__(self, connection: Connection):
        self.connection = connection
        self.dataToWrite = ""

    def flush(self):
        self.connection.send(self.dataToWrite)
        self.dataToWrite = ""

    def write(self, data: str):
        self.dataToWrite += data
        return len(data)


class WindowProcessingWorker:
    def __init__(self):
        self._outputRecv, self._outputSend = Pipe()

        self._outputText = ""

        self._loaded = Value("b", False)

        self._settingsRecvPipe, self._settingsSendPipe = Pipe()
        self._resultsRecvPipe, self._resultsSendPipe = Pipe()

        self._process = Process(target=self.Run,
                                args=(self._loaded, self._settingsRecvPipe, self._resultsSendPipe,
                                      self._outputSend),
                                daemon=True)
        self._process.start()

    def Results(self):
        if self.HasResults():
            return self._resultsRecvPipe.recv()

    def HasResults(self):
        return self._resultsRecvPipe.poll()

    def Process(self, settings: typing.List):
        self._outputText = ""
        self._settingsSendPipe.send(settings)

    @staticmethod
    def Run(loaded, settingsRecvPipe: Connection, resultsSendPipe: Connection,
            outputSendPipe: Connection):
        loaded.value = False
        import sys
        from Core.RunPipeline import WindowPipeline
        pipeWriter = PipeWriter(outputSendPipe)
        sys.stdout = pipeWriter
        sys.stderr = pipeWriter

        loaded.value = True
        settings = settingsRecvPipe.recv()
        window_name = settings[0]
        settings = settings[1:]
        
        loop = True
        with mss.mss() as sct:
            while loop:
                frame = capture_window(window_name)
                settings[1] = [frame]
                results = WindowPipeline(*settings)
                for res in results:
                    cv2.imshow('Classified Image', res)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        loop = False
                        break


    def ForceStop(self):
        if self._process.is_alive():
            self._process.kill()

    def GetOutputText(self):
        if not self._loaded.value:
            return "Loading OrganoID backend..."

        if self._outputRecv.poll():
            self._outputText += self._outputRecv.recv()

        return self._outputText



import subprocess
import pygame

class VideoWriter:
    def __init__(self, filename, resolution, fps):
        self.width, self.height = resolution
        self.fps = fps

        self.process = subprocess.Popen([
            'ffmpeg',
            '-y',  # overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgba',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
            '-an',  # no audio
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            filename
        ], stdin=subprocess.PIPE)

    def write_frame(self, surface):
        raw_bytes = pygame.image.tostring(surface, 'RGBA')
        self.process.stdin.write(raw_bytes)

    def close(self):
        self.process.stdin.close()
        self.process.wait()

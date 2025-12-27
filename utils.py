import os
import subprocess
import tempfile
import imageio_ffmpeg
from moviepy.editor import VideoFileClip

ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def extract_audio_from_video(video_path):
    dest_path = None
    try:
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        dest_path = temp_audio_path
        
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = 'ffmpeg'

        command = [
            ffmpeg_exe, '-y', '-i', video_path,
            '-ss', '00:00:00', '-t', '5',
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '22050', '-ac', '1',
            temp_audio_path
        ]
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_audio_path
    except Exception as e:
        if dest_path and os.path.exists(dest_path):
            os.unlink(dest_path)
        print(f"[WARNING] ffmpeg failed, falling back to moviepy: {e}")
        try:
            video = VideoFileClip(video_path)
            sub = video.subclip(0, min(5, video.duration))
            sub.audio.write_audiofile(temp_audio_path, logger=None)
            video.close()
            return temp_audio_path
        except Exception as e2:
            raise Exception(f"Error extracting audio: {str(e2)}")

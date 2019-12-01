import json
import os
from pytube import YouTube, exceptions
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import re

def read_audioset_ontology(path='ontology.json'):
    f = open(path, 'r')
    ontology = json.load(f)
    f.close()

    return ontology

def get_classes_info(ontology, num_classes=10):
    class_count = 0
    classes_info = []

    for entry in ontology:
        id, name, examples = entry['id'], entry['name'], entry['positive_examples']
        if len(examples) == 0:
            continue

        classes_info.append((id, name, examples))
        class_count += 1

        if class_count == num_classes:
            break

    return classes_info

def download_videos(class_info, output_prefix_path='AudioSet'):
    id, name, examples = class_info
    outpath = output_prefix_path + '/' + name

    if not os.path.exists(outpath):
        os.mkdir(outpath)

    for index, example in enumerate(examples):
        start_time = int(re.search('start=(.*)&', example).group(1))
        end_time = int(re.search('end=(.*)', example).group(1))

        videos_info = []
        try:
            yt = YouTube(example)
            stream = yt.streams.filter(file_extension='mp4').first()
            filepath = stream.download(outpath)
            videos_info.append((index, filepath, start_time, end_time))
        except:
            pass
        
        for video_info in videos_info:
            index, filepath, start_time, end_time = video_info
            targetpath = filepath.rsplit('/', 1)[0] + '/' + str(index) + '.mp4'
            ffmpeg_extract_subclip(filepath, start_time, end_time, targetname=targetpath)
            os.remove(filepath)

if __name__ == '__main__':
    folder = 'AudioSet'
    if not os.path.exists(folder):
        os.mkdir(folder)

    ontology = read_audioset_ontology()
    classes_info = get_classes_info(ontology)
    for class_info in classes_info:
        download_videos(class_info, output_prefix_path=folder)

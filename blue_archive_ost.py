from urllib.request import urlopen
from bs4 import BeautifulSoup
from tqdm import tqdm
from string import punctuation
from pydub import AudioSegment
from eyed3 import load
import os, json

source_url = 'https://bluearchive.fandom.com/wiki/Blue_Archive/Soundtrack'
dest_folder = 'blue_archive_ost/'
album = 'Blue Archive OST'

def download(url, folder='./', filename=None):
    if not folder.endswith('/'): folder += '/'
    if filename is None: filename = url[:url.find('?')].split('/')[-1]
    with open(folder + filename, 'wb') as dest:
        dest.write(urlopen(url).read())

def cap_first(string):
    return string[0].upper() + string[1:]

def rectify_title(string):
    stop_words = ['a', 'an', 'the', 'be', 'is', 'are', 'and', 'or', 'to', 'in', 'at', 'of', 'by', 'for', 'with']
    words = []
    for word in string.split():
        if word not in stop_words and word[0].islower():
            words.append(cap_first(word))
        else:
            words.append(word)
    string = cap_first(' '.join(words))
    if string[string.find('(')-1] != ' ': string = string.replace('(', ' (')
    string = string.replace('(All)', '').replace('(NEW!)', '').strip()
    return string

def rectify_filename(string):
    for punc in punctuation:
        string = string.replace(punc, '')
    return dest_folder + '_'.join([word.lower() for word in string.split()]) + '.mp3'

if not os.path.exists(dest_folder): os.mkdir(dest_folder)
print('Downloading source HTML...')
soup = BeautifulSoup(urlopen(source_url).read(), features='lxml')
print('Parsing webpage...')
playlist = soup.find('table', attrs={'class': 'wikitable'})
tracklist = []
for track in playlist.find_all('tr')[2:]:
    props = [prop.getText()[:-1] for prop in track.find_all('td')[:-2]]
    try:
        props.append(track.find('audio')['src'])
    except:
        continue
    else:
        try:
            props[0] = int(props[0])
        except:
            props.insert(0, tracklist[-1][0])
            if len(props[1]) < len(tracklist[-1][1]):
                del tracklist[-1]
                tracklist.append(props)
        else:
            tracklist.append(props)

print('Downloading %d tracks...' % len(tracklist))
metadata = []
for track_num, title_raw, artist, url in tqdm(tracklist):
    title = rectify_title(title_raw)
    filename = rectify_filename(title)
    download(url, dest_folder, 'temp.ogg')
    track = AudioSegment.from_ogg(dest_folder + 'temp.ogg')
    track.export(filename)
    os.remove(dest_folder + 'temp.ogg')
    track = load(filename)
    track.tag.track_num = track_num
    track.tag.title = title
    track.tag.artist = artist.replace(',', ';')
    track.tag.album = album
    track.tag.save()
    metadata.append({'track_num': track_num, 
                     'filename':  filename.split('/')[-1], 
                     'title':     title, 
                     'artist':    artist, 
                     'url':       url})
with open('metadata.json', 'w') as jsonFile:
    json.dump({'album': album, 'content': metadata}, jsonFile)
print('Done.')

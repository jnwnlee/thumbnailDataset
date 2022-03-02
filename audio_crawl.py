import requests
import pandas as pd
from tqdm import tqdm
import csv

#%%

df = pd.read_csv('./Context_tags_dataset.csv')
tags = ['car', 'car chill', 'chill', 'chill dance', 'chill night', 'chill party',
        'chill summer', 'chill work', 'club', 'dance', 'dance gym', 'dance party',
        'dance summer', 'gym', 'gym party', 'happy', 'night', 'party', 'party running',
        'party summer', 'party work', 'relax', 'running', 'sad', 'sleep', 'summer',
        'work', 'workout']
#df

#%%

song_ids = df['song_id'].tolist()
#song_ids

#%%

# import os
#
# for tag in tags:
#     os.mkdir('./audio2/'+tag)

#%%

no_audio_id = []

for idx, row in tqdm(df.iterrows()):
    song_id = row['song_id']
    try:
        r = requests.get('https://api.deezer.com/track/'+str(song_id)).json()
        audio_url = r.get('preview')
        if audio_url == '':
            print('song id '+str(song_id), 'No audio.')
            no_audio_id.append([str(song_id)])
            continue
        else:
            ext = audio_url.split('.')[-1]
            doc = requests.get(audio_url)
            # for tag in tags:
            #     if all(row[tag.split(' ')]==1):
            #         with open('/media/daftpunk2/home/seungheon/audio/'+tag+'/'+str(song_id)+'.'+ext, "wb") as f:
            #             f.write(doc.content)
            with open('/media/daftpunk2/home/seungheon/audio/' + str(song_id) + '.' + ext, "wb") as f:
                f.write(doc.content)
    except Exception as e:
        print('song id '+str(song_id), e)
        no_audio_id.append([str(song_id)])


with open('no_audio_id_list.csv', 'w', newline='') as f:
    write = csv.writer(f)
    write.writerows(no_audio_id)
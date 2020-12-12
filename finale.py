from __future__ import print_function
import random
from bs4 import BeautifulSoup
import requests
from typing import Collection    # (at top of module)
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import time
import sys
import pandas as pd
import tensorflow as tf
from tensorflow import keras


app = Flask(__name__)  # Initialize the flask App
model = keras.models.load_model('Neural Network')
data = pd.read_csv('song_data_combined_genre_label_final.csv')
data = data.drop(columns=['Track Name', 'Artist', 'Genre'])
data = data.drop(columns=['Label'])
data = data.astype('float')
# model = keras.models.load_model('Neural Network')  # loading the trained model


client_credentials_manager = SpotifyClientCredentials(client_id="5134ccf7425f474387886477436ab6c8",
                                                      client_secret="7d1ba68d2ae648b98420dd16d5f78c25")
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace = False
sp.trace = True

if len(sys.argv) > 1:
    tid = sys.argv[1]
else:
    tid = 'spotify:track:0FDzzruyVECATHXKHFs9eJ'  # spotify song URI

#start = time.time()
features = sp.audio_features(tid)
#delta = time.time() - start
# print(json.dumps(features))
#print("analysis retrieved in %.2f seconds" % (delta,))
document = json.dumps(features)
res = json.loads(document)
# converts str type to list
print("The converted dictionary : " + str(res))

# converts list type to dict, can access values now
l = res
res_dic = {}
for e in l:
    res_dic.update(e)
# print(res_dic)
# print(type(res_dic))
document = res_dic
# print(document)

keys = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

# get a filtered dictionary based on keys defined above
filtered_document = dict((k, document[k]) for k in keys if k in document)

# feature_extract = [document.get("danceability"), document.get("energy"), document.get("key"), document.get("loudness"), document.get("mode"), document.get(
#     "speechiness"), document.get("acousticness"), document.get("instrumentalness"), document.get("liveness"), document.get("valence"), document.get("tempo"), document.get("duration_ms"), document.get("time_signature")]


# def get_genre(track_name, artist):
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0'}
#     link = f'https://www.last.fm/music/{artist}/_/{track_name}'
#     try:
#         req = requests.get(link, headers=headers, timeout=None)
#         soup = BeautifulSoup(req.content, features="html.parser")
#         tags = ""
#         for tag in soup.find_all(attrs={'class': 'tag'}):
#             href = tag.find('a').get('href')
#             href = href.replace("/tag/", "")
#             tags += href+", "
#         tags = tags.rstrip(", ")
#         # print(f"Tags for {track_name} are {tags}")
#         return tags
#     except Exception as e:
#         print("not found")
#         print(e)


# track_name = 'Binding Light'
# artist = 'The Weeknd'
# tags = get_genre(track_name, artist)
# # print(tags)
# lst = tags.split()
# # print(lst)
def get_genre_spotify(artist_name, sp):
    try:
        result = sp.search(artist_name)
        track = result['tracks']['items'][0]

        artist = sp.artist(track["artists"][0]["external_urls"]["spotify"])
        genre_list = artist["genres"]
        tags = ''
        for genre in genre_list:
            tags = tags + genre + ','
        tags = tags.rstrip(",")
        print(tags)
    except Exception as e:

        pass
    return tags


def get_genre(track_name, artist):
    client_id = '90d4bf0d722b4e6892fedd3eb7dea15d'
    client_secret = 'b85dd00354944ae4a37e1b1aa7871f5d'
    client_credentials_manager = SpotifyClientCredentials(
        client_id, client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0'}
    link = f'https://www.last.fm/music/{artist}/_/{track_name}'
    try:
        req = requests.get(link, headers=headers, timeout=None)
        soup = BeautifulSoup(req.content, features="html.parser")
        tags = ""
        for tag in soup.find_all(attrs={'class': 'tag'}):
            href = tag.find('a').get('href')
            href = href.replace("/tag/", "")
            tags += href+","
        print(track_name)

        tags = tags.rstrip(",")
        if tags == '':
            print('spotify')
            tags = get_genre_spotify(track_name, sp)

        return tags
    except Exception as e:
        print("not found")
        print(e)


def clean_genre(genre):
    genre = genre.lower()
    genre = genre.replace("+", "")
    genre = genre.replace("-", "")
    genre = genre.replace(" ", "")
    return genre


def label_genre(song_genre):
    top_genre_list = ['hiphop', 'rap', 'pop', 'electronic',
                      'trap', 'indie', 'rnb', 'rock', 'dance', 'reggaeton']

    song_genre = clean_genre(song_genre)
    song_genre_list = song_genre.split(',')  # convert string to list

    song_genre_label = {}

    print("song_genre", song_genre_list)
    for top_genre in top_genre_list:
        if any(top_genre in song_genre for song_genre in song_genre_list):
            # prevent miss classify "trap" as "rap" since "rap" in "trap" is True
            if top_genre == 'rap' and 'rap' not in song_genre_list:
                #print(top_genre, genre_list)
                song_genre_label[top_genre] = 0
                continue
            song_genre_label[top_genre] = 1

        else:
            song_genre_label[top_genre] = 0
    return song_genre_label


# genre = 'rnb,POP,lana+del re-y,2016,dance pop,rnb,electDADSSDonic,canadian,trap'
# lst = ' '.join([str(elem) for elem in lst])
# genre = lst
# song_genre = label_genre(genre)
# print(song_genre)
# print(feature_extract)
# print(document)
# print(filtered_document)

# 2-step process to combine the two dictionaries
# compiled_dict = filtered_document.copy()
# compiled_dict.update(song_genre)

# print(compiled_dict)


@app.route('/')  # Homepage
def home():
    import pandas as pd
    data = pd.read_csv('song_data_combined_genre_label_final.csv')
    columns = data.columns.values
    print(columns)
    return render_template('index.html', )


def get_url(track_name, artist):
    track_name = track_name
    search_result = sp.search(track_name)

    print(f"Total {len(search_result['tracks']['items'])} result found: ")
    for i in range(len(search_result['tracks']['items'])):
        print(
            f'################### Rearch result {i} #############################')
        URL = search_result['tracks']['items'][i]['external_urls']['spotify']
        print('song URL\n', URL)

        artists_list = []
        artists_dic = search_result['tracks']['items'][i]['artists']
        print(artists_dic)
        for artist_info in artists_dic:
            artists_list.append(artist_info['name'])
        print('artists:\n', artists_list)

        track_features = sp.audio_features(URL)[0]
        print('Track features:\n', track_features)

        if artist in artists_list:
            return URL


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    # retrieving values from form
    user_input = [x for x in request.form.values()]
    print(user_input)
    track_name = user_input[0]
    artist = user_input[1]

    genre = get_genre(track_name, artist)
    print("genre", genre)
    #final_features = [np.array(init_features)]

    genre_dic = label_genre(genre)
    print(genre_dic)
    # prediction = model.predict(final_features)  # making prediction

    URL = get_url(track_name, artist)
    print("[URL]:", URL)
    track_features = sp.audio_features(URL)[0]
    print("[track_features]: ", track_features)

    columns = ['Track Name', 'Artist', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
               'tempo', 'duration_ms', 'time_signature', 'Genre', 'hiphop', 'rap', 'pop',
               'electronic', 'trap', 'indie', 'rnb', 'rock', 'dance', 'reggaeton', 'Label']

    compiled_dict = track_features.copy()
    compiled_dict.update(genre_dic)
    print(compiled_dict)

    # this is a df of user input data
    input_data = pd.DataFrame(compiled_dict, index=[0])
    # print(data)

    input_data = input_data.drop(
        columns=['type', 'id', 'uri', 'track_href', 'analysis_url'])
    print(input_data)

    # take a random example for now. This will be the actual user data

    new_data = (input_data-data.min())/(data.max()-data.min())
    new_data['bias'] = 1

    print(new_data)

    result = model.predict(new_data)
    print("[PREDICTION]: ", result)

    # rendering the predicted result
    return render_template('index.html', prediction_text='Probability of being Hit Song: {}'.format(result[0][0]))


if __name__ == '__main__':
    app.run(debug=True)

from bs4 import BeautifulSoup
from requests_html import HTMLSession
import pandas as pd
import numpy as np
import pretty_midi
import requests
import difflib
import os
import re

# YEARS = [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018]
YEARS = [2015, 2017, 2018]
def get_webpage(year):
    # session = HTMLSession()
    # print("Connecting...")
    # page = 'https://www.piano-e-competition.com/midi_'+str(year)+'.asp'
    # html_page = session.get(page)
    # print("Get webpage info.")
    html_page = open("data_competition/html/"+str(year)+" MIDI files.html", encoding='ISO-8859-1')
    html_page = BeautifulSoup(html_page, features="lxml")
    return html_page

def get_performer_info(html_page, index):
    performer_info = html_page.find_all(string=re.compile("Nationality:"))
    performer = performer_info[index].parent.parent
    performer = " ".join(performer.text.split('\n'))
    performer = performer.split('Nationality:')
    performer_name = performer[0].split(" ")
    performer_name = list(filter(lambda x: x!= '', performer_name))
    performer_name = " ".join(performer_name).split(":")[1][1:]
    performer_nation = performer[1][1:]
    return performer_name, performer_nation

def generate_and_download():
    records = open("records.txt", "a+")
    if os.path.isfile("meta.csv"):
        meta = pd.read_csv("meta.csv")
        n = meta.shape[0]
    else:
        meta = pd.DataFrame(columns=['canonical_composer', 'canonical_title', 'year', 'round', 'performer', 'nationality', 'duration', 'filepath'])
        n = 0
    for i in YEARS:
        savepath = 'data_maestro/'+str(i)+"/"
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
            
        html_page = get_webpage(i)
        composition_info = html_page.find_all("table", class_="detail-text")
        for j in range(len(composition_info)):
            performer_name, performer_nation = get_performer_info(html_page, j) 
            print(performer_name)  
            components = composition_info[j].find_all("tr")
            for k in range(len(components)):
                children = components[k].find_all("td")
                if not children:
                    continue
                if children[0].find("b") or children[0].find("strong"):
                    perform_round = children[0].string
                    continue
                if (children[0].string) and (len(children[0].string) == 1) and (len(children[1].text) <= 1):
                    continue
                else:
                    if (children[0].string) and (len(children[0].string) > 1) :
                        # print("-----Have----")
                        canonical_composer = children[0].string
                        canonical_composer = canonical_composer.split(" ")
                        canonical_composer = list(filter(lambda x: x!= '', canonical_composer))
                        canonical_composer = " ".join(canonical_composer)
                    elif perform_round == meta['round'].iloc[n-1]:
                        # print("----Same----")
                        canonical_composer = meta['canonical_composer'].iloc[n-1]
                    else:
                        # print("-----None-----")
                        canonical_composer = ''
                    

                    if children[1].string:
                        canonical_title = children[1].string
                    else:
                        canonical_title = children[1].text
                    
                    if ("\n" in canonical_title) or ("\t" in canonical_title) or ("\r" in canonical_title) or ("    " in canonical_title):
                        canonical_title = " ".join(canonical_title.split("\n"))
                        canonical_title = " ".join(canonical_title.split("\t"))
                        canonical_title = " ".join(canonical_title.split("\r"))
                        canonical_title = " ".join(canonical_title.split("    "))
                        canonical_title = canonical_title.split(" ")
                        canonical_title = list(filter(lambda x: x!= '', canonical_title))
                        canonical_title = " ".join(canonical_title)
                    
                    if not children[1].a:
                        continue
                    link = children[1].a['href']
                    filepath = savepath+performer_name.split(' ')[-1]+'_'+link.split('/')[-1]
                    if not os.path.isfile(filepath):
                        print(canonical_title)
                        try:
                            filedata = requests.get(link, stream=True)
                        except:
                            records.write(link + "\n")
                            continue

                        with open(filepath, 'wb') as f:
                            f.write(filedata.content)

                    try:
                        midi_data = pretty_midi.PrettyMIDI(filepath)
                    except:
                        records.write(link + "\n")
                        continue
                    duration = midi_data.get_end_time()
                    meta.loc[n] = [canonical_composer, canonical_title, i, perform_round, 
                                performer_name, performer_nation, duration, filepath]
                    n += 1
                    
        meta.to_csv('meta.csv', index=False)

def match(origin_csv, new_csv):
    new_data = pd.read_csv(new_csv)
    origin_data = pd.read_csv(origin_csv)
    # for i in range(origin_data.shape[0]):
    count = 0
    none = 0
    null = np.sum(pd.isnull(origin_data['performer']))
    print(null)
    for i in range(origin_data.shape[0]):
        if not (pd.isnull(origin_data.iloc[i]['performer'])):
            continue
        time = origin_data.iloc[i]['duration']
        year = origin_data.iloc[i]['year']
        title = origin_data.iloc[i]['canonical_title']
        composer = origin_data.iloc[i]['canonical_composer']
        # print(title)
        # print(origin_data.iloc[i])
        select = new_data[new_data['year'] == year]
        # similarity_title = []
        # for i in range(select.shape[0]):
            # s = difflib.SequenceMatcher("None", select, title)
        # print(composer)
        # print(select['canonical_composer'])
        similarity_composer = np.asarray([difflib.SequenceMatcher(None, x, composer).ratio() for x in select['canonical_composer']])
        similarity_title = np.asarray([difflib.SequenceMatcher(None, x, title).ratio() for x in select['canonical_title']])
        
        # print(select[similarity_title == np.max(similarity_title)])
        # print(select[similarity_composer == np.max(similarity_composer)])


        select = select[(similarity_title == 1)]
        # & (similarity_composer == np.max(similarity_composer))]
        
        mv = select.shape[0]
        # print(mv)
        # for j in range(select.shape[0]):
        #     print(select.iloc[j])
        if mv == 1:
            count += 1
            origin_data['performer'].iloc[i] = select['performer'].iloc[0]
            origin_data['nationality'].iloc[i] = select['nationality'].iloc[0]
        if mv == 0:
            none += 1
    print("one: %d" % count)
    print("none: %d" % none)
    print("multiple: %d" % (null - count - none))
    origin_data.to_csv(origin_csv, index=False)

match("data_maestro/maestro-v3.0.0.csv", "data_competition/meta.csv")

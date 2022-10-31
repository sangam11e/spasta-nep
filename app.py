from urllib import request
from flask import Flask,render_template,url_for,jsonify,request
from flask_wtf import FlaskForm
from wtforms import TextAreaField,SubmitField
import pandas as pd
import time

app=Flask(__name__)
#Spell Checker's code
import time
corpus = []

def loadCorpus():
    #function to load the dictionary/corpus and store it in a global list
    global corpus
    # path12='D:\Spasta Nepali data+files\dictionary.txt'
    path12='D:/Spasta Nepali data+files/project on/added_s/unique_words.txt'
    with open(path12,'r', encoding="utf-8") as csv_file:
        corpus =csv_file.readlines()#csv.reader(csv_file)
        # for line in corpus:
        #     corpus.append(line[1])
    # corpus=corpus[:160000]
    # return corpus
           
loadCorpus()       
def getLevenshteinDistance(s, t):

    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i

    for i in range(1, cols):
        dist[0][i] = i
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
                
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution

    return dist[row][col]

def getCorrectWord(word):
    min_dis=100
    correct_word=""
    for s in corpus:
        cur_dis = getLevenshteinDistance(s,word)
        if min_dis > cur_dis :
            min_dis = cur_dis
            correct_word = s
    return correct_word
def processInput():
    inputtext=input('Enter text')   
    words=inputtext.strip().split()
    output=''
     
    for word in words:
            
                if word not in corpus :
                    corrected= getCorrectWord(word)
                    output = output + corrected + ' '
                else:
                    output = output + word + ' '
    print(output)

#Spell Cheker ends



stop1=time.time()
print(start,stop1," Time elapsed is : ",stop1-start)


@app.route("/")
def index():
    return render_template("ajax.html",)

@app.route("/test_ajax1",methods=["POST","GET"])
def testing():
    global temp_A
    if request.method=="POST":
        data2=request.get_json(force=True)
        print(data2['page_data'])
        text21=data2['page_data']
       
        return jsonify(data2['page_data'])



if __name__=="main":
    app.run(debug=True)

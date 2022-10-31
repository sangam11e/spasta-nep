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


#Grammar Checker Starts

# from nepali_stemmer.stemmer import NepStemmer
# nepstem = NepStemmer()
start=time.time()
x=["cs.txt","gc_books.txt","gc_newspaper__extract.txt","gc_webtext__extract.txt"]
def reading(filename):
  with open(filename,"r",encoding="utf-8") as f:
    x=f.read().split("\n")
    return x


temp_path="D:/Spasta Nepali data+files/project on/added_s/"
temp0=reading(temp_path+x[0])
temp1=reading(temp_path+x[1])
temp1=temp1[:len(temp1)-3]
temp2=reading(temp_path+x[2])
temp2=temp2[:len(temp2)-3]
temp3=reading(temp_path+x[3])
temp3=temp3[:len(temp3)-3]
all_files_list=temp0+temp1+temp2+temp3
del temp1
del temp0
del temp2
del temp3
print(len(all_files_list))
def return_tuples(list1):
  temp_list=[]
  # print(len(list1))
  for ii in list1:
    temp1=ii.split("\t")
    key1=temp1[0].strip()
    # print(temp1)
    key2=temp1[1].split("\n")[0].strip()
    # print(key2)
    temp2=(key1,key2)
    # print(type(temp2),temp2,temp1[0].strip())
    temp_list.append(temp2)
  return temp_list

get_tuples=return_tuples(all_files_list)

def get_frequency(tup1):

    freqs = {} # dictionary to be returned
    curr_words = set()
    ### START CODE HERE ###
    for tup in tup1:
        word, label = tup
        if word not in curr_words:
            new_dict = {}
            new_dict[label] = 1
            freqs[word] = new_dict
        else:
            cur_dict = freqs[word]
            if label in set(cur_dict.keys()):
                cur_dict[label] += 1
            else:
                cur_dict[label] = 1
            freqs[word] = cur_dict
        curr_words.add(word)
    ### END CODE HERE ###
    return freqs
freqs=get_frequency(get_tuples)
get_all_tags=reading(temp_path+"/111_tags.txt")
all_tags=[]
for _ in get_all_tags:
  temp=_.split("\t")[0].strip()
  if temp=='*' or temp=='':
    continue
  all_tags.append(temp)

del get_all_tags

import numpy as np
def transition_count(tup2):
  transition={}
  for _ in range(len(tup2)-1):
    key1=tup2[_][1]
    key2=tup2[_+1][1]
    key3=(key1,key2)
    # print(key3)
    temp_dict1={}
    if key3 not in transition.keys():
      transition[key3]=1
    else:
      transition[key3]+=1
  return transition

temp1_a=np.zeros((111,111))
temp_A=pd.DataFrame(temp1_a,index=all_tags, columns =all_tags)


get_transition_count=transition_count(get_tuples)
ttt=0
def create_transition_matrix(constant_val,tags_with_count,count_transition):
  global temp_A
  all_tags=list(tags_with_count.keys())#sorted
  A=np.zeros((len(all_tags),len(all_tags)))
  tuple1=set(count_transition.keys())
  for i in range(len(all_tags)):
    for j in range(len(all_tags)):
      count=0
      # if(all_tags[i]=="*"):
      #   continue
      temp_tuple1=(all_tags[i],all_tags[j])
      if temp_tuple1 in count_transition.keys():
        count=count_transition[temp_tuple1]
      count_prev_tag=tags_with_count[all_tags[i]]
      temp_calc=(count + constant_val) / (count_prev_tag + constant_val * len(all_tags))
    #   A[i,j] = temp_calc
      temp_A.loc[all_tags[i],all_tags[j]]=temp_calc

#   return A


def emission_count(tup2):
  emission={}
  set1=set()
  for _ in tup2:
    temp1=_
    key1=(temp1[0].strip(),temp1[1].strip())    
    if key1 not in emission.keys():
      emission[key1]=1
    else:
      emission[key1]+=1
  return emission

tag_with_count={}
ij=0
# print(len(all_files_list))
for i in all_files_list:
  ij+=1
  a=i.split("\t")[1].split("\n")[0]
  if a not in tag_with_count:
    if a=="*" or a=="":
      continue
    tag_with_count[a]=1
  else:
    tag_with_count[a]+=1

emission_counts=emission_count(get_tuples)
create_transition_matrix(0,tag_with_count,get_transition_count)
def create_emission_matrix(constant_val,tags_count,emission_count_data,vocab):
  all_tags=list(tags_count.keys())
  print(all_tags[:20])
  B=np.zeros((len(all_tags),len(emission_count_data)))
  # B_temp=pd.DataFrame(B,index=all_tags,columns=)
  for i in range(len(all_tags)):
    for j in range(len(emission_count_data)):
      count=0
      tuple1=(vocab[j],all_tags[i])
      if tuple1 in emission_count_data.keys():
        count=emission_count_data[tuple1]
      count_tag=tags_count[all_tags[i]]

      B[i,j] = (count +constant_val) / (count_tag +constant_val * len(vocab))
  return B

only_text=[]
get_only_text=reading(temp_path+"/only_all_text.txt")
B=create_emission_matrix(0.00,tag_with_count,emission_counts,get_only_text)
words_name=[word for word,tag in emission_counts.keys()]
B_sub = pd.DataFrame(B,index=list(tag_with_count.keys()), columns =words_name)#pd.DataFrame(A, index=all_tags, columns = all_tags )

def Viterbi(sentence_list,state):
  global temp_A
  global B_sub
  global freqs
  p=[]
  for words in sentence_list:
    # print(len(freqs[words]))
    for j in freqs[words].keys():
      # print(j,words,freqs[words])
      transition_p=temp_A.loc[state[-1],j]
      emission_p=B_sub.loc[j,words][0]
      state_p=transition_p*emission_p
      # print(emission_p)
      p.append(state_p)
    print(max(p))
  return list(freqs[words].keys())[p.index(max(p))]

def sentence_checker(sents):
  temp_pos=[]
  for x in sents:
    # temp_pos.append(list(x[1].keys())[0])
    temp_pos.append(x[1])
  return temp_pos

def stemmer(combined_words):
  global temp_path
  path2=temp_path+"/suffix.txt"
  with open(path2,"r",encoding="utf-8") as f:
    suffixes=f.readlines()
  modified_suffixes=[y.split("|")[0] for y in suffixes]
  sep_words=[]
  for x in modified_suffixes:
    if x in combined_words:
      temp_list=[combined_words.split(x)[0],x]
      if temp_list[0] in freqs.keys():
        sep_words.append(temp_list)
        break
  return(sep_words)

def check_unique_POS(token1,indx=0):
    # print(type(token1))
    # print(len(freqs[token1]))
    if token1 in freqs.keys() :#and len(freqs[token1])==1:
        # print(freqs[token1],len(freqs[token1])," : 1")
        key_list=list(freqs[token1].keys())
        value_list=list(freqs[token1].values())
        # print(key_list,value_list,max(value_list))
        temp_tuple=(token1,key_list[value_list.index(max(value_list))])
        # temp_tuple=(token1,freqs[token1])
    else:
        temp_list=stemmer(token1)
        # print(len(freqs[token1]))
        return temp_list
        # temp_tuple=(token1,freqs[token1])
    return temp_tuple

def pos_tag(texts):
    texts=texts.replace("|","।")
    sents=texts.split("।")
    print(f"There are {len(sents)-1} sentences in the provided texts")
    words=[sents[y].split() for y in range(len(sents)-1)]
    count=0
    tagging=[]
    state=[]
    for y in words:
        temp_tagging=[]
        for each_word in y:
            if each_word in freqs.keys():
                # print(freqs[each_word])
                # if 
                key_list=list(freqs[each_word].keys())
                value_list=list(freqs[each_word].values())
                # print(key_list,value_list,max(value_list))
                temp_tuple=(each_word,key_list[value_list.index(max(value_list))])
                temp_tagging.append(temp_tuple)
                # print(temp_tuple)
                print("temp tuple 1st  ",temp_tuple,type(temp_tuple[1]))
                count+=1
            else:
                import nltk
                
                new_words=nltk.tokenize.word_tokenize(each_word)
                # print(new_words)
                for new_tokens in new_words:
                    temp_tuple=check_unique_POS(new_tokens)
                    if type(temp_tuple)!=list:
                      temp_tagging.append(temp_tuple)
                      # print(len(freqs[new_tokens])," :  second")
                    else:
                      # print(temp_tuple)
                      for xy in temp_tuple:
                        #  print("asdsad ",xy,type(xy))
                         for p in xy:
                           temp_tuple=check_unique_POS(p)
                          #  print("temp tuple 2nd ",temp_tuple)
                           count+=1
                           if type(temp_tuple)!=list:
                              temp_tagging.append(temp_tuple)
                              # tagging.append(temp_tagging)
                         

        tagging.append(temp_tagging)
        # print(count,tagging)
    # only_tags=[]
    # for _ in tagging:
    #   for aa in _:
    #     temp22=Viterbi(_,state)
    #     state.append(temp22)

    return tagging



def return_probab(pos_list):
  global temp_A
  probab=[]
  dict_probab={}
  # print(type(pos_list))
  for temp_pos_list in pos_list:
    temp_pos_list.insert(0,"--s--")
    temp_pos_list.append("YF")
    # print("temp_pos ",temp_pos_list)
    # temp_probab=A_sub.loc[]
  
  for y in pos_list:
    for x in range(len(y)-1):
      key1=(y[x],y[x+1])
      # print(freqs[key1])
      temp_calc=temp_A.loc[y[x],y[x+1]]#A_sub.iloc[list22.index(y[x]),list22.index(y[x+1])]
      # print(y[x],",",y[x+1],temp_calc,"   ",A_sub.iloc[list22.index(y[x]),list22.index(y[x+1])]," ",A_sub.iloc[list22.index(y[x]),list22.index(y[x+1])]==temp_calc)
      dict_probab[key1]=temp_calc
      probab.append(temp_calc)
  print(dict_probab)
  return dict_probab
  # print(max(probab))

  # print(pos_list)

del all_files_list
del temp1_a
del get_only_text
del words_name



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
        tagging=pos_tag(text21)
        
        get_pos=[]
        print("Tagging",tagging)
        for i in tagging:  
            get_pos.append(sentence_checker(i))
        print(get_pos,type(get_pos))

        dict2=return_probab(get_pos)
        print(dict2)
        return jsonify(data2['page_data'])



if __name__=="main":
    app.run(debug=True)
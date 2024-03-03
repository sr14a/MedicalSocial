from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import matplotlib.pyplot as plt
import re
import cv2
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from string import punctuation
import os
from nltk.tokenize import word_tokenize
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import random

load_index = 0
global svm_classifier

filename = []
word_vector = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return cleanPost(string.strip().lower())

with open('dataset/question.json', "r") as f:
    lines=f.readlines()
    for line in lines:
        arr = line.split("#")
        cleanedLine = clean_str(arr[0])
        cleanedLine = cleanedLine.strip()
        cleanedLine = cleanedLine.lower()
        word_vector.append(cleanedLine)
        filename.append(arr[1])
    f.close()        

stopwords=stopwords = nltk.corpus.stopwords.words("english")
tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords,use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
tfidf = tfidf_vectorizer.fit_transform(word_vector).toarray()        
df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
print(str(df))
print(df.shape)
df = df.values
X = df[:, 0:df.shape[1]]
X = np.asarray(X)
filename = np.asarray(filename)
word_vector = np.asarray(word_vector)



def Admin(request):
    if request.method == 'GET':
       return render(request, 'Admin.html', {})
       
def AdminLogin(request):
    if request.method == 'POST':
      username = request.POST.get('username', False)
      password = request.POST.get('password', False)
      if username == 'admin' and password == 'admin':
       context= {'data':'welcome '+username}
       return render(request, 'AdminScreen.html', context)
      else:
       context= {'data':'login failed'}
       return render(request, 'Admin.html', context)

def test(request):
    if request.method == 'GET':
       return render(request, 'test.html', {})

def ChatData(request):
    if request.method == 'GET':
        question = request.GET.get('mytext', False)
        cleanedLine = clean_str(question)
        cleanedLine = cleanedLine.strip()
        cleanedLine = cleanedLine.lower()
        testArray = []
        testArray.append(cleanedLine)
        testStory = tfidf_vectorizer.transform(testArray).toarray()
        similarity = 0
        user_story = 'Sorry! I am not trained for given question'
        print(testStory.shape)
        testStory = testStory[0]
        for i in range(len(X)):
            classify_user = dot(X[i], testStory)/(norm(X[i])*norm(testStory))
            if classify_user > similarity and classify_user > 0.50:
                similarity = classify_user
                user_story = filename[i]
        print(question+" "+user_story)
        return HttpResponse(user_story, content_type="text/plain")


def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})
       
def Accuracy(request):
    if request.method == 'GET':
       return render(request, 'Accuracy.html', {})

def Login(request):
    if request.method == 'GET':
       return render(request, 'Login.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})

def ChangePassword(request):
    if request.method == 'GET':
       return render(request, 'ChangePassword.html', {})

def PostTopic(request):
    if request.method == 'GET':
       return render(request, 'PostTopic.html', {})

def HomePage(request):
    if request.method == 'GET':
        user = ''
        with open("session.txt", "r") as file:
            for line in file:
                user = line.strip('\n')
        status_data = ''
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == user:
                    status_data = row[5]
                    break
            if status_data == 'none':
                status_data = ''   
            output = ''
            output+='<table border=0 align=center width=100%><tr><td><img src=/static/profiles/'+user+'.png width=200 height=200></img></td>'
            output+='<td><font size=3 color=black>'+status_data+'</font></td><td><font size=3 color=black>welcome : '+user+'</font></td></tr></table></br></br>'
            output+=getPostData()
            context= {'data':output}
            return render(request, 'UserScreen.html', context)

def getRating(pid):
    rating = 0
    count = 0
    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select * FROM comment")
        rows = cur.fetchall()
        for row in rows:
            post = row[0]
            if post == pid:
                rating = rating + float(row[3])
                count = count + 1
    if count > 0:
        rating = rating/count
    return str(rating)

def getPostData():
    output = '<table border=1 align=center>'
    output+='<tr><th><font size=3 color=black>Username</font></th>'
    output+='<th><font size=3 color=black>Image</font></th>'
    output+='<th><font size=3 color=black>Image Name</font></th>'
    output+='<th><font size=3 color=black>Name</font></th>'
    output+='<th><font size=3 color=black>Topic</font></th>'
    output+='<th><font size=3 color=black>Description</font></th>'
    output+='<th><font size=3 color=black>Overall Rating</font></th>'
    output+='<th><font size=3 color=black>View Post</font></th></tr>'

    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select * FROM post")
        rows = cur.fetchall()
        for row in rows:
            username = row[0]
            post_id = str(row[1])
            image = row[2]
            name = row[3]
            topic = row[4]
            description = row[5]
            output+='<tr><td><font size=3 color=black>'+username+'</font></td>'
            output+='<td><img src=/static/post/'+post_id+'.png width=200 height=200></img></td>'
            output+='<td><font size=3 color=black>'+image+'</font></td>'
            output+='<td><font size=3 color=black>'+name+'</font></td>'
            output+='<td><font size=3 color=black>'+topic+'</font></td>'
            output+='<td><font size=3 color=black>'+description+'</font></td>'
            output+='<td><font size=3 color=black>'+getRating(post_id)+'</font></td>'
            output+='<td><a href=\'PostComment?id='+post_id+'\'><font size=3 color=black>Click Here</font></a></td></tr>'
    output+="</table><br/><br/><br/><br/><br/><br/>"        
    return output

def getComments(pid):
    output = '<table border=1 align=center>'
    output+='<tr><th><font size=3 color=black>Post ID</font></th>'
    output+='<th><font size=3 color=black>Username</font></th>'
    output+='<th><font size=3 color=black>Comment</font></th>'
    output+='<th><font size=3 color=black>Rating</font></th></tr>'
    

    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select * FROM comment where post_id='"+pid+"'")
        rows = cur.fetchall()
        for row in rows:
            pid = row[0]
            username = str(row[1])
            comment = row[2]
            rate = row[3]
            output+='<tr><td><font size=3 color=black>'+pid+'</font></td>'
            output+='<td><font size=3 color=black>'+username+'</font></td>'
            output+='<td><font size=3 color=black>'+comment+'</font></td>'
            output+='<td><font size=3 color=black>'+rate+'</font></td></tr>'
            
    output+="</table><br/><br/><br/><br/><br/><br/>"        
    return output

def PostMyComment(request):
    global load_index
    global svm_classifier
    if request.method == 'POST':
        comment = request.POST.get('comment', False)
        pid = request.POST.get('pid', False)
        user = ''
        with open("session.txt", "r") as file:
            for line in file:
                user = line.strip('\n')
        if load_index == 0:
            svm_classifier = joblib.load('svmClassifier.pkl')
            load_index = 1
        X =  [comment]
        sentiment = svm_classifier.predict(X)
        senti = sentiment[0]
        rate = 0
        if senti == 0:
            rate = random.randint(0,2)
        if senti == 1:
            rate = random.randint(3,5)
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO comment(post_id,username,comment,rate) VALUES('"+pid+"','"+str(user)+"','"+comment+"','"+str(rate)+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        output = '<table align=\"center\" width=\"80\">'
        output+= '<tr><td><b>Comment</b></td><td><input type=\"text\" name=\"comment\" style=\"font-family: Comic Sans MS\" size=\"60\"></td></tr>'
        output+= '<tr><td></td><td><input type=\"hidden\" name=\"pid\" style=\"font-family: Comic Sans MS\" value='+pid+'></td></tr>'
        output+= '<tr><td></td><td><input type=\"submit\" value=\"Submit\"></td></tr></table><br/><br/>'
        output+= getComments(pid)
        context= {'data':output}
        return render(request, 'PostCommentPage.html', context)
    

def PostComment(request):
    if request.method == 'GET':
        pid = request.GET['id']
        output = '<table align=\"center\" width=\"80\">'
        output+= '<tr><td><b>Comment</b></td><td><input type=\"text\" name=\"comment\" style=\"font-family: Comic Sans MS\" size=\"60\"></td></tr>'
        output+= '<tr><td></td><td><input type=\"hidden\" name=\"pid\" style=\"font-family: Comic Sans MS\" value='+pid+'></td></tr>'
        output+= '<tr><td></td><td><input type=\"submit\" value=\"Submit\"></td></tr></table><br/><br/>'
        output+= getComments(pid)
        context= {'data':output}
        return render(request, 'PostCommentPage.html', context)


def PostMyTopic(request):
    if request.method == 'POST':
        name = request.POST.get('name', False)
        topic = request.POST.get('topic', False)
        description = request.POST.get('description', False)
        myfile = request.FILES['image']
        imagename = request.FILES['image'].name
        user = ''
        with open("session.txt", "r") as file:
            for line in file:
                user = line.strip('\n')
        count = 0

        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select count(*) FROM post")
            rows = cur.fetchall()
            for row in rows:
                count = row[0]
        count = count + 1        

        fs = FileSystemStorage()  
        filename = fs.save('D:/MedicalSocial/User/static/post/'+str(count)+'.png', myfile)
      
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "INSERT INTO post(username,post_id,image,name,topic,description) VALUES('"+user+"','"+str(count)+"','"+imagename+"','"+name+"','"+topic+"','"+description+"')"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record Inserted")
        status_data = ''
        if db_cursor.rowcount == 1:
            con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
            with con:
                cur = con.cursor()
                cur.execute("select * FROM register")
                rows = cur.fetchall()
                for row in rows:
                    if row[0] == user:
                        status_data = row[5]
                        break
            if status_data == 'none':
                status_data = ''   
            output = ''
            output+='<table border=0 align=center width=100%><tr><td><img src=/static/profiles/'+user+'.png width=200 height=200></img></td>'
            output+='<td><font size=3 color=black>'+status_data+'</font></td><td><font size=3 color=black>welcome : '+user+'</font></td></tr></table></br></br>'
            output+=getPostData()
            context= {'data':output}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Error in post topic'}
            return render(request, 'PostTopic.html', context)
    
    
def ChangeMyPassword(request):
    if request.method == 'POST':
        password = request.POST.get('password', False)
        user = ''
        with open("session.txt", "r") as file:
            for line in file:
                user = line.strip('\n')
                        
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "update register set password='"+password+"' where username='"+user+"'"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record updated")
        status_data = ''
        if db_cursor.rowcount == 1:
            con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
            with con:
                cur = con.cursor()
                cur.execute("select * FROM register")
                rows = cur.fetchall()
                for row in rows:
                    if row[0] == user and row[1] == password:
                        status_data = row[5]
                        break
            if status_data == 'none':
                status_data = ''   
            output = ''
            output+='<table border=0 align=center width=100%><tr><td><img src=/static/profiles/'+user+'.png width=200 height=200></img></td>'
            output+='<td><font size=3 color=black>'+status_data+'</font></td><td><font size=3 color=black>welcome : '+user+'</font></td></tr></table></br></br>'
            output+=getPostData()
            context= {'data':output}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Error in updating status'}
            return render(request, 'UpdateStatus.html', context)      

def UpdateStatus(request):
    if request.method == 'GET':
       return render(request, 'UpdateStatus.html', {})

def UpdateMyStatus(request):
    if request.method == 'POST':
        status = request.POST.get('status', False)
        user = ''
        with open("session.txt", "r") as file:
            for line in file:
                user = line.strip('\n')
                        
        db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
        db_cursor = db_connection.cursor()
        student_sql_query = "update register set status='"+status+"' where username='"+user+"'"
        db_cursor.execute(student_sql_query)
        db_connection.commit()
        print(db_cursor.rowcount, "Record updated")
        if db_cursor.rowcount == 1:
            output = ''
            output+='<table border=0 align=center width=100%><tr><td><img src=/static/profiles/'+user+'.png width=200 height=200></img></td>'
            output+='<td><font size=3 color=black>'+status+'</font></td><td><font size=3 color=black>welcome : '+user+'</font></td></tr></table></br></br>'
            output+=getPostData()
            context= {'data':output}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Error in updating status'}
            return render(request, 'UpdateStatus.html', context)  

def EditProfile(request):
    if request.method == 'GET':
        output = ''
        user = ''
        with open("session.txt", "r") as file:
            for line in file:
                user = line.strip('\n')
        output = ''
        username = ''
        password = ''
        contact = ''
        email = ''
        address = ''
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM register where username='"+user+"'")
            rows = cur.fetchall()
            for row in rows:
                username = row[0]
                password = row[1]
                contact = row[2]
                email = row[3]
                address = row[4]
        output+='<tr><td><b>Username</b></td><td><input type=text name=username style=font-family: Comic Sans MS size=30 value='+username+' readonly></td></tr>'
        output+='<tr><td><b>Password</b></td><td><input type=password name=password style=font-family: Comic Sans MS size=30 value='+password+'></td></tr>'
        output+='<tr><td><b>Contact&nbsp;No</b></td><td><input type=text name=contact style=font-family: Comic Sans MS size=20 value='+contact+'></td></tr>'
        output+='<tr><td><b>Email&nbsp;ID</b></td><td><input type=text name=email style=font-family: Comic Sans MS size=40 value='+email+'></td></tr>'
        output+='<tr><td><b>Address</b></td><td><input type=text name=address style=font-family: Comic Sans MS size=60 value='+address+'></td></tr>'
        context= {'data':output}
        return render(request, 'EditProfile.html', context)    

def Signup(request):
    if request.method == 'POST':
      username = request.POST.get('username', False)
      password = request.POST.get('password', False)
      contact = request.POST.get('contact', False)
      email = request.POST.get('email', False)
      address = request.POST.get('address', False)
      myfile = request.FILES['image']

      fs = FileSystemStorage()
      filename = fs.save('D:/MedicalSocial/User/static/profiles/'+username+'.png', myfile)
      
      db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
      db_cursor = db_connection.cursor()
      student_sql_query = "INSERT INTO register(username,password,contact,email,address,status) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"','none')"
      db_cursor.execute(student_sql_query)
      db_connection.commit()
      print(db_cursor.rowcount, "Record Inserted")
      if db_cursor.rowcount == 1:
       context= {'data':'Signup Process Completed'}
       return render(request, 'Register.html', context)
      else:
       context= {'data':'Error in signup process'}
       return render(request, 'Register.html', context)

def EditMyProfile(request):
    if request.method == 'POST':
      username = request.POST.get('username', False)
      password = request.POST.get('password', False)
      contact = request.POST.get('contact', False)
      email = request.POST.get('email', False)
      address = request.POST.get('address', False)
      myfile = request.FILES['image']

      if os.path.exists('D:/MedicalSocial/User/static/profiles/'+username+'.png'):
          os.remove('D:/MedicalSocial/User/static/profiles/'+username+'.png')

      fs = FileSystemStorage()
      filename = fs.save('D:/MedicalSocial/User/static/profiles/'+username+'.png', myfile)
      
      db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
      db_cursor = db_connection.cursor()
      student_sql_query = "update register set username='"+username+"',password='"+password+"',contact='"+contact+"',email='"+email+"',address='"+address+"' where username='"+username+"'"
      db_cursor.execute(student_sql_query)
      db_connection.commit()
      print(db_cursor.rowcount, "Record updated")
      status_data = ''
      if db_cursor.rowcount == 1:
          con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
          with con:
              cur = con.cursor()
              cur.execute("select * FROM register")
              rows = cur.fetchall()
              for row in rows:
                  if row[0] == username and row[1] == password:
                      status_data = row[5]
                      break
          if status_data == 'none':
              status_data = ''            
          output = ''
          output+='<table border=0 align=center width=100%><tr><td><img src=/static/profiles/'+username+'.png width=200 height=200></img></td>'
          output+='<td><font size=3 color=black>'+status_data+'</font></td><td><font size=3 color=black>welcome : '+username+'</font></td></tr></table></br></br>'
          output+=getPostData()
          context= {'data':output}
          return render(request, 'UserScreen.html', context)
      else:
       context= {'data':'Error in editing profile'}
       return render(request, 'EditProfile.html', context)    
        
def UserLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        status = 'none'
        status_data = ''
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    status = 'success'
                    status_data = row[5]
                    break
        if status_data == 'none':
            status_data = ''
        if status == 'success':
            file = open('session.txt','w')
            file.write(username)
            file.close()
            output = ''
            output+='<table border=0 align=center width=100%><tr><td><img src=/static/profiles/'+username+'.png width=200 height=200></img></td>'
            output+='<td><font size=3 color=black>'+status_data+'</font></td><td><font size=3 color=black>welcome : '+username+'</font></td></tr></table></br></br>'
            output+=getPostData()
            context= {'data':output}
            return render(request, 'UserScreen.html', context)
        if status == 'none':
            context= {'data':'Invalid login details'}
            return render(request, 'Login.html', context)

def ViewUsers(request):
    if request.method == 'GET':
       strdata = '<table border=1 align=center width=50%><tr><th>User Name</th><th>Email</th><th>Address</th><th>Contact</th><th>Image</th></tr><tr>'
       con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
       with con:
          cur = con.cursor()
          cur.execute("select * FROM register")
          rows = cur.fetchall()
          for row in rows: 
             strdata+='<td>'+row[0]+'</td><td>'+row[3]+'</td><td>'+str(row[4])+'</td><td>'+str(row[2])+'</td><td><img src=/static/profiles/'+row[0]+'.png width=200 height=200></img></td></tr>'
    context= {'data':strdata}
    return render(request, 'Viewusers.html', context)
            
def ViewUser(request):
    if request.method == 'GET':
        user = ''
        with open("session.txt", "r") as file:
            for line in file:
                user = line.strip('\n')
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM register where username='"+user+"' ")
            rows = cur.fetchall()
            for row in rows: 
               output=''
               output= '<table border=1 align=center width=50%><tr><th>User Name</th><th>Email</th><th>Address</th><th>Contact</th><th>Image</th></tr><tr>'
               output+='<td>'+row[0]+'</td><td>'+row[3]+'</td><td>'+str(row[4])+'</td><td>'+str(row[2])+'</td>'
               output+='<td><img src=/static/profiles/'+user+'.png width=200 height=200></img></td></tr></table></br></br>'
               context= {'data':output}
               return render(request, 'Viewuser.html', context)
               
def Viewposts(request):
    if request.method == 'GET':
       output=''
       output+=getPost()
       context= {'data':output}
       return render(request, 'Viewposts.html', context)
       
       
def getPost():
    output = '<table border=1 align=center>'
    output+='<tr><th><font size=3 color=black>Username</font></th>'
    output+='<th><font size=3 color=black>Image</font></th>'
    output+='<th><font size=3 color=black>Image Name</font></th>'
    output+='<th><font size=3 color=black>Name</font></th>'
    output+='<th><font size=3 color=black>Topic</font></th>'
    output+='<th><font size=3 color=black>Description</font></th>'
    output+='<th><font size=3 color=black>Overall Rating</font></th>'
    output+='<th><font size=3 color=black>View Comment</font></th></tr>'

    con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'MedicalSocial',charset='utf8')
    with con:
        cur = con.cursor()
        cur.execute("select * FROM post")
        rows = cur.fetchall()
        for row in rows:
            username = row[0]
            post_id = str(row[1])
            image = row[2]
            name = row[3]
            topic = row[4]
            description = row[5]
            output+='<tr><td><font size=3 color=black>'+username+'</font></td>'
            output+='<td><img src=/static/post/'+post_id+'.png width=200 height=200></img></td>'
            output+='<td><font size=3 color=black>'+image+'</font></td>'
            output+='<td><font size=3 color=black>'+name+'</font></td>'
            output+='<td><font size=3 color=black>'+topic+'</font></td>'
            output+='<td><font size=3 color=black>'+description+'</font></td>'
            output+='<td><font size=3 color=black>'+getRating(post_id)+'</font></td>'
            output+='<td><a href=\'ViewComment?id='+post_id+'\'><font size=3 color=black>Click Here</font></a></td></tr>'
    output+="</table><br/><br/><br/><br/><br/><br/>"        
    return output


def ViewComment(request):
    if request.method == 'GET':
        pid = request.GET['id']
        output=''
        output+= getComments(pid)
        context= {'data':output}
        return render(request, 'ViewComment.html', context)
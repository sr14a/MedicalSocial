<p>Medical Text classification here uses NLTK, DJango, Python for its full fledged working of the web interface in the web page. </p><br>
<p>Chat bot application clears the medical related queries and for that chat bot we a classifier that is, NLTK. </p><br>
<p>the process is quite interesting in here. The steps it follows in classifying the text gives us full clarity of how this chatbot works.</p><br>
<ol>
<li>NLTK-Natural Language Tool Kit</li>
<li>Tokens</li>
<li>Stop Words</li>
<li>Stemming + Lemmatization</li>
<li>Combinations</li>
</ol>
<br>
<p>These steps are followed while processing the text. <br>
NLTK, the common text processing library in python, it provides the user with different interfaces which help in the text classification, tokenizing, stemming, etc.<br>
Whatever the text that bot is receiving, it will divide that whole text into multiple tokens, and it analyses the specific words that describe the query given.<br>
Stop words are those which, the classifier removes the excess small verbal or simple grammar terms (in, or, is, those, the.. etc) and the sentence simple and easy to understand<br>
Stemming and Lemmatization, are now come over and analyse those specific text, and classify from the data set which we will train the system, by some test and train data.<br>
The words which are classified from the query are combined with the date set and the query give the answer with which we have trained by train data.<br>
<br>
</p>
This model can be retrained with the users own questions also. for that, we can write our own query in the questions data, then we tarin the model with the commands: 
" python manage.py makemigrations" & " python manage.py migrate" <br>
these commands will successfully train our model with new set of data. 
<br>

<br>
to run this project, enter the command : <br>
"python manage.py runserver" in command prompt in the same directory where the project folder is located. 
<br>
<h2> This whole project gives a web application where a user can post any type of medicla related data and also ask queries to chatbot</h2>

import firebase_admin
from firebase_admin import credentials
from google.cloud import storage, firestore
import os
import re
from sentence_transformers import SentenceTransformer, util
from flask import Flask,render_template,redirect,url_for,request,jsonify,send_file,Response
import requests
from flask_cors import CORS
import json

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Gov Chatbot/govchatbot-firebase-adminsdk-zrkrx-24d2a117de.json"

app = Flask(__name__)
CORS(app) 

project_id = "govchatbot" 
db = firestore.Client(project=project_id)
cred = credentials.Certificate('Gov Chatbot/govchatbot-firebase-adminsdk-zrkrx-24d2a117de.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'govchatbot.appspot.com' })
storage_client = storage.Client()

#filename stores the document name to be downloaded, messages stores the list of response and query of chatbot, pdf_blob stores the pdf file in blob format to send
#flag is used to add the extra "yes" message
filename=None
messages=[]
pdf_blob=None
flag=0

# Section of code to store user feedbacks
@app.route('/feedback', methods=['POST'])
def feedback():
    global messages
    data = request.get_json()
    feedback = data.get('feedback')
    index = data.get('index')
    try:
        feedback_ref = db.collection('feedback')
        feedback_doc = feedback_ref.document()
        feedback_doc.set({
           'user_response':messages[index-1],
           
            'chatbot_response': messages[index],
            'feedback': feedback})
        print(f'Feedback stored with ID: {feedback_doc.id}')
        response = {
        'status': 'success',
        'message': 'Feedback received successfully'
        }
        return jsonify(response)
    except Exception as e:
        print(f'An error occurred: {e}')
        response = {
        'status': 'failed',
        'message': 'Sending Failed'
        }
        return jsonify(response)


def retrieve_and_display_feedback():
    try:
        feedback_ref = db.collection('feedback').get()
        for doc in feedback_ref:
            print(f'Document ID: {doc.id}')
            print(f'User Response: {doc.get("user_response")}')
            print(f'Chatbot Response: {doc.get("chatbot_response")}')
            print(f'Feedback Text: {doc.get("feedback")}')
            print('---')
    except Exception as e:
        print(f'An error occurred: {e}')


#Section of code which is used to retreive documents from dataset
model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
model_accuracy = SentenceTransformer(model_name)
questions_dictionary = {
    "Domicile": [
        "ഡോമിസൈൽ സർട്ടിഫിക്കറ്റിന് എന്തെല്ലാം രേഖകൾ വേണം",
        "ഡോമിസൈൽ സർട്ടിഫിക്കറ്റ് നേടുന്നതിനുള്ള രേഖകൾ എന്തൊക്കെ",
        "ഡോമിസൈൽ സർട്ടിഫിക്കറ്റിന് ആവശ്യമുള്ള രേഖകൾ ഏതൊക്കെ"
    ],
    "Income": [
        "വരുമാന സർട്ടിഫിക്കറ്റിന് എന്തെല്ലാം രേഖകൾ വേണം",
        "വരുമാന സർട്ടിഫിക്കറ്റ് നേടുന്നതിനുള്ള രേഖകൾ എന്തൊക്കെ",
        "വരുമാന സർട്ടിഫിക്കറ്റിന് ആവശ്യമുള്ള രേഖകൾ ഏതൊക്കെ"
    ],
    "Land-ownership": [
        "ഭൂ ഉടമസ്ഥാവകാശ സർട്ടിഫിക്കറ്റിന് എന്തെല്ലാം രേഖകൾ വേണം",
        "ഭൂ ഉടമസ്ഥാവകാശ സർട്ടിഫിക്കറ്റ് നേടുന്നതിനുള്ള രേഖകൾ എന്തൊക്കെ",
        "ഭൂ ഉടമസ്ഥാവകാശ സർട്ടിഫിക്കറ്റിന് ആവശ്യമുള്ള രേഖകൾ ഏതൊക്കെ"
    ],
    "Marriage": [
        "വിവാഹ സർട്ടിഫിക്കറ്റിന് എന്തെല്ലാം രേഖകൾ വേണം",
        "വിവാഹ സർട്ടിഫിക്കറ്റ് നേടുന്നതിനുള്ള രേഖകൾ എന്തൊക്കെ",
        "വിവാഹ സർട്ടിഫിക്കറ്റിന് ആവശ്യമുള്ള രേഖകൾ ഏതൊക്കെ"
    ],
    "Nativity": [
        "നേറ്റിവിറ്റി സർട്ടിഫിക്കറ്റിന് എന്തെല്ലാം രേഖകൾ വേണം",
        "നേറ്റിവിറ്റി സർട്ടിഫിക്കറ്റ് നേടുന്നതിനുള്ള രേഖകൾ എന്തൊക്കെ",
        "നേറ്റിവിറ്റി സർട്ടിഫിക്കറ്റിന് ആവശ്യമുള്ള രേഖകൾ ഏതൊക്കെ"
    ],
    "Non-creamy": [
        "നോൺ-ക്രീമി ലെയർ സർട്ടിഫിക്കറ്റിന് എന്തെല്ലാം രേഖകൾ വേണം",
        "നോൺ-ക്രീമി ലെയർ സർട്ടിഫിക്കറ്റ് നേടുന്നതിനുള്ള രേഖകൾ എന്തൊക്കെ",
        "നോൺ-ക്രീമി ലെയർ സർട്ടിഫിക്കറ്റിന് ആവശ്യമുള്ള രേഖകൾ ഏതൊക്കെ"
    ]
}

def calculate_similarity(a, b):
    actual_embeddings = model_accuracy.encode(a, convert_to_tensor=True)
    predicted_embeddings = model_accuracy.encode(b, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(actual_embeddings, predicted_embeddings)
    return cos_sim

def download_pdf_from_storage(bucket_name, file_name):
    global pdf_blob
    try:
        bucket = storage_client.bucket(bucket_name)
        pdf_blob = bucket.blob(file_name)
    except Exception as e:
        print(f"An error occurred: {e}")

def search_and_download_documents(user_input):
    global filename,flag
    best_match = {"certificate": None, "similarity": 0}
    for certificate, questions in questions_dictionary.items():
        for question in questions:
            similarity = calculate_similarity(user_input, question)
            if similarity > best_match["similarity"]:
                best_match = {"certificate": certificate, "similarity": similarity}
    print(best_match)
    #best similary sentence is found and its corresponding certificate name is used to retreieve the file
    if best_match["similarity"] > 0.95:
        certificate_name = best_match["certificate"]
        regex_pattern = f"^{certificate_name}.*"
        storage_bucket = storage_client.bucket("govchatbot.appspot.com")
        blobs = storage_bucket.list_blobs()
        for blob in blobs:
            if re.match(regex_pattern, blob.name):  
                filename=blob.name
                print(filename)
                flag=1
                download_pdf_from_storage("govchatbot.appspot.com", blob.name)
    else:
        print("No matching document found.")

@app.route('/get_certificate', methods=['GET'])
def get_blob():
    global filename, pdf_blob,messages
    if filename:
        try:
            #Download blob content into memory
            blob_data = pdf_blob.download_as_bytes()
            messages.append(f'Document {filename}')
            messages.append('Document Download')
            #Sending file to react as pdf
            return Response(blob_data, mimetype='application/pdf')
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'error': 'Error generating blob.'}), 500
    else:
        return jsonify({'error': 'Filename parameter missing.'}), 400


@app.route('/answer', methods=['POST'])
def answer():
    data = request.get_json()
    global messages,flag
    url='http://127.0.0.1:1111/receive_data'
    #searched if documents are needed by user by checking his user query with some pre-defined questions
    search_and_download_documents(data['question'])
    response=requests.post(url,json=data)
    result=response.json()  
    #to add the yes message at the end of answer(when document requested)
    if flag == 1:
        download = " നിങ്ങൾക്ക് ആവശ്യമായ എല്ലാ രേഖകളും ഡൗൺലോഡ് ചെയ്യണമെങ്കിൽ അതെ എന്ന് ടൈപ്പ് ചെയ്യുക"
        result['answer']=result['answer']+download
    #converted to dict format because it already is in json
    response = {'answer': result['answer']}
    messages.append(data['question'])
    messages.append(result['answer'])
    flag = 0
    #returns answer as json format
    return app.response_class(response=json.dumps(response), mimetype='application/json')


if __name__ == '__main__':
    app.run(port=5000,debug=True)
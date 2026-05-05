import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

firebase_key = os.getenv("FIREBASE_KEY")

if firebase_key:
    cred_dict = json.loads(firebase_key)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
else:
    raise Exception("Firebase key not found")

db = firestore.client()
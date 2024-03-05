from fastapi import FastAPI
from transformers import MarianMTModel, MarianTokenizer
import os
import torch
from os import path

app = FastAPI()

@app.get("/CMSAI/isMarianMTModelAvailable")
def isMarianMTModelAvailable():
    modelFolder = path.relpath("models/MarianMT")
    if(os.path.isdir(modelFolder)): 
        # en-fr is to convert English to French, can dynamically manipulate this  
        model_name = "Helsinki-NLP/opus-mt-en-fr"            
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
       
        tokenizer.save_pretrained(modelFolder)
        model.save_pretrained(modelFolder)
        return True
    else:
        return False    

@app.post("/CMSAI/TranslateText")
def translateTextUsingMarianMT(input:str):
    modelFolder = path.relpath("models/MarianMT")
    tokenizer = MarianTokenizer.from_pretrained(modelFolder)
    model = MarianMTModel.from_pretrained(modelFolder,local_files_only= True)

    inputs = tokenizer(input, return_tensors = 'pt')
   
    #In this case, we don't need to do any training hence no gradient needs to be updated during back propogation
    #This will not keep any track of gradients and will just ensure that we do forward propagation
    with torch.no_grad():
        outputs = model.generate(**inputs)

    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

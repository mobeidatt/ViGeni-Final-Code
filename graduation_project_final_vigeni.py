
pip install moviepy transformers torch pillow
pip install SpeechRecognition
pip install AudioSegment
pip install openai
pip install openai fpdf requests


import speech_recognition as sr
import moviepy as mp
import os
from pydub import AudioSegment
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline


video_path = "c:/Users/20200373/Desktop/The Bird and the Whale — US English accent (TheFableCottage.com) (1).mp4"

# Extract audio from video
clip = mp.VideoFileClip(video_path)
clip.audio.write_audiofile(r"converted.wav")

# Load the audio file with pydub 
audio = AudioSegment.from_wav("converted.wav")

chunk_length_ms = 30 * 1000  # 30 seconds
chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

r = sr.Recognizer()

full_result = ""
for i, chunk in enumerate(chunks):
    chunk_path = f"chunk_{i}.wav"
    chunk.export(chunk_path, format="wav")

    with sr.AudioFile(chunk_path) as source:
        audio_file = r.record(source)
        try:
            result = r.recognize_google(audio_file)
            full_result += result + " "
        except sr.UnknownValueError:
            print(f"Chunk {i}: Google Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            print(f"Chunk {i}: Could not request results from Google Speech Recognition service; {e}")

    os.remove(chunk_path)

with open('recognized.txt', mode='w') as file:
    file.write(full_result)
    print("ready!")

"""##Summrized Text

User Choses if arabic or english
"""

text = ""
with open("recognized.txt", "r") as file:
    text = file.read()


summarizer = pipeline("summarization")

original_length = max(1, int(len(text.split()) * 0.3))
summary = summarizer(text, max_length=original_length, min_length=int(original_length*0.8), do_sample=False)

print(summary)


translated=[]
chunks=[]
language=input("Enter the Language you want (ar or en):")

if(language=='ar'):
    model_path = "C:/Users/20200373/Desktop/TranslationViGeni"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    words=text.split()
    spliting_size=int(len(text)/128)
    for i in range(0, len(words), spliting_size):
        chunk = " ".join(words[i:i+spliting_size])
        chunks.append(chunk)
    for i in chunks:
        input_text = i
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated.append(decoded_output)
    text=" ".join(translated)

else:
    print("")

'''
from transformers import pipeline

summarizer = pipeline("summarization")

original_length = max(1, int(len(text.split()) * 0.3))
summary = summarizer(text, max_length=original_length, min_length=int(original_length*0.8), do_sample=False)

print(summary)'''

summary_text = summary[0]["summary_text"]

len(summary_text.split())

if len(summary_text.split()) < 100:
  current_loops = 3 #0-99
elif len(summary_text.split()) < 200:
  current_loops = 4 #100-199
elif len(summary_text.split()) < 300:
  current_loops = 5 #200-299
elif len(summary_text.split()) < 400:
  current_loops = 6 #300-399
elif len(summary_text.split()) < 600 and len(summary_text.split()) >= 400:
  current_loops = 7 #400-599
elif len(summary_text.split()) < 800 and len(summary_text.split()) >= 600:
  current_loops = 8 #600-799
elif len(summary_text.split()) < 1000 and len(summary_text.split()) >= 800:
  current_loops = 9 #800-999
elif len(summary_text.split()) >= 1000:
  current_loops = 10 #1000+

int(len(summary_text.split())/current_loops)

" ".join(summary_text.split()[0:10])

li = []
group_size = len(summary_text.split()) // current_loops  # Calculate the size of each group

for i in range(0, len(summary_text.split()), group_size):
    if len(li) == current_loops - 1:
        li.append(" ".join(summary_text.split()[i:]))
        break
    li.append(" ".join(summary_text.split()[i:i + group_size]))

# Output the groups
print(li)

len(li)

"""from nltk.tokenize import sent_tokenize


# Split the text into sentences
prompts = sent_tokenize(summary_text)

for i, prompt in enumerate(prompts, ):
    print(prompt)"""


"""# **Create A Title for the Story**"""

final_string = "create title image for this prompts " + summary_text
final_string

"""# Title image"""

from openai import OpenAI
from IPython.display import Image, display
import requests
import time

client = OpenAI(api_key="API KEY")


response = client.images.generate(
model="dall-e-3",
prompt=final_string,
size="1024x1024",
quality="standard",n=1,
    )

image_url = response.data[0].url
print(final_string)

image_data = requests.get(image_url).content
file_name = "Title_image_.png"
with open(file_name, "wb") as image_file:
     image_file.write(image_data)

display(Image(url=image_url))

time.sleep(10)  

print("Images saved successfully.")


"""# Content images"""

client = OpenAI(api_key="API KEY")

z = 0
for i in li:
    response = client.images.generate(
        model="dall-e-3",
        prompt=i,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    print(li[z])

    image_data = requests.get(image_url).content
    file_name = f"generated_image_{z}.png"
    with open(file_name, "wb") as image_file:
        image_file.write(image_data)

    display(Image(url=image_url))

    z += 1
    time.sleep(10)  # Pause to respect the rate limit

print("Images saved successfully.")

from IPython.display import Image, display

num_images = z  # Replace z with the actual number of images saved

for i in range(num_images):
    file_name = f"generated_image_{i}.png"

splited=text.split(" ")
numOfWords=len(splited)/len(li)
numOfWords=int(numOfWords)
numOfWords

if(language=='en'):
    with open('recognized.txt', 'r') as file:
      words = file.read().split()
    chunks = []
    for i in range(0, len(words), numOfWords):
      chunk = " ".join(words[i:i+numOfWords])
      chunks.append(chunk)
else:
    chunks = []
    splited_text=text.split()
    for i in range(0, len(splited_text), numOfWords):
      chunk = " ".join(splited_text[i:i+numOfWords])
      chunks.append(chunk)

client = OpenAI(api_key="API KEY")

if(language=='en'):
    response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=f"Create a title for the story based on the following summary:\n{summary_text}",
    max_tokens=17
    )
else:
    response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=f"اصنع عنوان للقصة بالعربي بناءا على الملخص فيما يلي:\n{summary_text}",
    max_tokens=17
    )

title = response.choices[0].text.strip()
title

pip install arabic-reshaper

pip install python-bidi

from fpdf import FPDF
import os
import arabic_reshaper
from bidi.algorithm import get_display

# PDF Initialization
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# File paths
logo_path = "C:/Users/20200373/Desktop/logooooo (1) (1).png"  
title_image_path = "Title_image_.png"

pdf.add_font('Amiri', '', 'C:/Users/20200373/Desktop/Amiri-Bold.ttf', uni=True)
pdf.add_font('Amiri', 'B', 'C:/Users/20200373/Desktop/Amiri-Bold.ttf', uni=True)

def apply_page_theme(pdf):
    pdf.set_fill_color(230, 240, 255)
    pdf.rect(0, 0, 210, 297, 'F')  

    pdf.set_draw_color(100, 130, 180)  
    pdf.set_line_width(1.5)
    pdf.rect(12, 12, 186, 273)  

    # Add the logo
    pdf.image(logo_path, x=170, y=10, w=25) 

# Reshape and prepare Arabic text
reshaped = arabic_reshaper.reshape(title)  
title = get_display(reshaped)

# Title Page
pdf.add_page()
apply_page_theme(pdf)

pdf.set_font("Amiri", "B", size=18)  
pdf.set_text_color(40, 70, 120)  
pdf.set_xy(10, 50)  
pdf.multi_cell(0, 10, txt=title, align="C")  

# Add the title image
pdf.image(title_image_path, x=35, y=80, w=140) 

for w, chunk in enumerate(chunks):
    if w == len(chunks) - 1 and len(chunk.split()) < 20:
        print(f"Skipping the last page because the text has less than 20 words.")
        continue

    # Check if the corresponding image exists
    file_name = f"generated_image_{w}.png"
    if os.path.exists(file_name):
        pdf.add_page()
        apply_page_theme(pdf)

        # Add the image
        pdf.image(file_name, x=31, y=33, w=150, h=145)  

        reshaped_chunk = arabic_reshaper.reshape(chunk)  
        bidi_chunk = get_display(reshaped_chunk)  

        pdf.set_font('Amiri', '', 12)  
        pdf.set_text_color(0, 0, 0)  
        pdf.set_xy(25, 180)  

        if language == 'ar':
            pdf.multi_cell(163, 10, txt=bidi_chunk, align="R")  
        else:
            pdf.multi_cell(163, 10, txt=bidi_chunk, align="L")  
    else:
        print(f"Image file {file_name} not found. Skipping this page.")

# Save the PDF
pdf.output("C:/Users/20200373/Desktop/storybook_arabic.pdf")
print("Your Story is Ready!!")


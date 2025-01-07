
pip install moviepy transformers torch pillow

pip install SpeechRecognition

"""# Vid to Text"""

pip install AudioSegment

import speech_recognition as sr
import moviepy as mp
import os
from pydub import AudioSegment

# Define video_path
video_path = "C:/Users/20200373/Desktop/The Bird and the Whale — US English accent (TheFableCottage.com) (1).mp4"

# Extract audio from video
clip = mp.VideoFileClip(video_path)
clip.audio.write_audiofile(r"converted.wav")

# Load the audio file with pydub (can handle large audio files more efficiently)
audio = AudioSegment.from_wav("converted.wav")

# Split the audio into chunks (e.g., 30 seconds per chunk)
chunk_length_ms = 30 * 1000  # 30 seconds
chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

# Initialize recognizer
r = sr.Recognizer()

# Process each chunk separately and append the results
full_result = ""
for i, chunk in enumerate(chunks):
    # Export the chunk to a temporary file
    chunk_path = f"chunk_{i}.wav"
    chunk.export(chunk_path, format="wav")

    # Recognize the speech in the chunk
    with sr.AudioFile(chunk_path) as source:
        audio_file = r.record(source)
        try:
            result = r.recognize_google(audio_file)
            full_result += result + " "
        except sr.UnknownValueError:
            print(f"Chunk {i}: Google Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            print(f"Chunk {i}: Could not request results from Google Speech Recognition service; {e}")

    # Clean up temporary chunk file
    os.remove(chunk_path)

# Exporting the result to a text file
with open('recognized.txt', mode='w') as file:
    file.write(full_result)
    print("ready!")

"""##Summrized Text

User Choses if arabic or english
"""

text=" "
with open("recognized.txt", "r") as file:
    text = file.read()

from transformers import pipeline

summarizer = pipeline("summarization")

original_length = max(1, int(len(text.split()) * 0.3))
summary = summarizer(text, max_length=original_length, min_length=int(original_length*0.8), do_sample=False)

print(summary)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

translated_list=[]
language=input("Enter the Language you want (ar or en):")

if(language=='ar'):
    split_128=int(len(text)/128)
    splited_text=text.split()
    chunks = [splited_text[i:i + split_128] for i in range(0, len(splited_text), split_128)]
    text= [' '.join(chunk) for chunk in chunks]
    model_path = "C:/Users/20200373/Desktop/TranslationViGeni"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for i in text:
      input_text = i
      inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
      outputs = model.generate(**inputs)
      decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
      translated_list.append(decoded_output)
    text = " ".join(translated_list)
else:
    print("")

text

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
    # Adjust the last group to include any remaining words
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

pip install openai

pip install openai fpdf requests

"""# **Create A Title for the Story**"""

final_string = "create title image for this prompts " + summary_text

# Print the result
final_string

"""# Title image"""

from openai import OpenAI
from IPython.display import Image, display
import requests
import time
# Initialize OpenAI client
client = OpenAI(api_key="API KEY")


response = client.images.generate(
model="dall-e-3",
prompt=final_string,
size="1024x1024",
quality="standard",n=1,
    )

    # Get the image URL
image_url = response.data[0].url
print(final_string)

    # Download the image and save it locally
image_data = requests.get(image_url).content
file_name = "Title_image_.png"
with open(file_name, "wb") as image_file:
     image_file.write(image_data)

    # Display the image in the notebook
display(Image(url=image_url))

time.sleep(10)  # Pause to respect the rate limit

print("Images saved successfully.")

# To download the files, you can use the following line:

"""# Content images"""

# Initialize OpenAI client
client = OpenAI(api_key="API KEY")

z = 0
for i in li:
    # Generate image from prompt
    response = client.images.generate(
        model="dall-e-3",
        prompt=i,
        size="1024x1024",
        quality="standard",
        n=1,
    )

    # Get the image URL
    image_url = response.data[0].url
    print(li[z])

    # Download the image and save it locally
    image_data = requests.get(image_url).content
    file_name = f"generated_image_{z}.png"
    with open(file_name, "wb") as image_file:
        image_file.write(image_data)

    # Display the image in the notebook
    display(Image(url=image_url))

    z += 1
    time.sleep(10)  # Pause to respect the rate limit

print("Images saved successfully.")

from IPython.display import Image, display

# Number of images you saved
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
logo_path = "C:/Users/20200373/Desktop/logooooo (1) (1).png"  # Ensure this file exists in your environment
title_image_path = "Title_image_.png"

# Add Arabic font (Ensure you have the TTF font file in the correct directory)
pdf.add_font('Amiri', '', 'C:/Users/20200373/Desktop/Amiri-Bold.ttf', uni=True)
pdf.add_font('Amiri', 'B', 'C:/Users/20200373/Desktop/Amiri-Bold.ttf', uni=True)

# Function to apply the theme to every page
def apply_page_theme(pdf):
    # Set the background color for the page
    pdf.set_fill_color(230, 240, 255)  # Light blue background color
    pdf.rect(0, 0, 210, 297, 'F')  # Fill the entire page with the background color

    # Add a decorative frame
    pdf.set_draw_color(100, 130, 180)  # Soft blue frame color
    pdf.set_line_width(1.5)
    pdf.rect(12, 12, 186, 273)  # Draw a slightly inset frame

    # Add the logo
    pdf.image(logo_path, x=170, y=10, w=25)  # Adjust x, y, and w for position and size

# Reshape and prepare Arabic text
reshaped_text = arabic_reshaper.reshape(arabic_text)  # Reshape the Arabic text
bidi_text = get_display(reshaped_text)  # Reorder the text to properly display from right to left
reshaped = arabic_reshaper.reshape(title)  # Reshape the Arabic text
title = get_display(reshaped)
# Title Page
pdf.add_page()
apply_page_theme(pdf)

# Add the title text with smaller font and nicer style
pdf.set_font("Amiri", "B", size=18)  # Smaller size and nicer font (Times Bold)
pdf.set_text_color(40, 70, 120)  # Soft blue text color
pdf.set_xy(10, 50)  # Position the title
pdf.multi_cell(0, 10, txt=title, align="C")  # Center the title text

# Add the title image
pdf.image(title_image_path, x=35, y=80, w=140)  # Adjust x, y, and w for positioning

# List of chunks to loop through


# Loop through the chunks and add them to the PDF
for w, chunk in enumerate(chunks):
    # Reshape and reorder the chunk for Arabic text
    reshaped_chunk = arabic_reshaper.reshape(chunk)  # Reshape the Arabic text
    bidi_chunk = get_display(reshaped_chunk)  # Reorder the text to properly display from right to left

    # Add a new page for each chunk
    pdf.add_page()
    apply_page_theme(pdf)  # Apply the theme to this page

    # Add a space for the image or extra padding if there's no image
    pdf.ln(10)

    # Check if the image exists (example: generated_image_0.png, generated_image_1.png, etc.)
    file_name = f"generated_image_{w}.png"
    if os.path.exists(file_name):
        # Center the image on the page with extra padding for a "storybook" layout
        pdf.image(file_name, x=31, y=33, w=150, h=145)  # Adjust position and size as needed
    else:
        # Add placeholder text for pages without images (optional)
        pdf.set_font("Arial", "I", size=15)
        pdf.set_text_color(100, 100, 100)
        pdf.set_xy(25, 120)
        pdf.multi_cell(0, 10, txt="No image available for this section.", align="C")

    # Add a text area for the chunk
    pdf.set_font('Amiri', '', 12)  # Use the Amiri Arabic font
    pdf.set_text_color(0, 0, 0)  # Black color for text
    pdf.set_xy(25, 180)  # Position the text starting point

    # Use the reshaped and bidi text for Arabic content
    if (language=='ar'):
        pdf.multi_cell(163, 10, txt=bidi_chunk, align="R")  # Right alignment for Arabic text
    else:
        pdf.multi_cell(163, 10, txt=bidi_chunk, align="L")  # Right alignment for Arabic text
# Save the decorated storybook PDF to a file
pdf.output("C:/Users/20200373/Desktop/storybook_arabic_with_images.pdf")
print("Your Story is Ready!!")


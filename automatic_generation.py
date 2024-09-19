import tkinter as tk
from tkinter import *
from tkinter import font
from PIL import ImageTk, Image
from music21 import *
import glob
from tqdm import tqdm
import numpy as np
import random
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
import os

# Funcția pentru a citi fișierele MIDI
def read_files(file):
    notes = []
    notes_to_parse = None
    midi = converter.parse(file)
    instrumente = instrument.partitionByInstrument(midi)

    for part in instrumente.parts:
        if 'Piano' in str(part):        # Preia doar notele de pian
            notes_to_parse = part.recurse()

        for element in notes_to_parse:     # Imparte acordurile in note
            if type(element) == note.Note:
                notes.append(str(element.pitch))
            elif type(element) == chord.Chord:
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

# Funcția pentru generarea muzicii
def generate_music():
    global entry
    global file_path

    user_input = entry.get()    # Inputul primit de la utilizator
    file_path = "All Midi Files/" + user_input

    label.configure(text=f"Your {user_input} music file is ready!", bg='#f5f4ae')

    if not file_path:
        label.configure(text="Please enter a directory first!", bg='#f5f4ae')
        return

    all_files = glob.glob(file_path + '/*.mid')
    notes_array = np.array([read_files(i) for i in tqdm(all_files, position=0, leave=True)], dtype=object)

    # Verificăm dacă modelul există deja pe disc
    if os.path.exists("ModeL"):
        # Note unice
        note_muzicale = sum(notes_array, [])
        unique_notes = list(set(note_muzicale))

        # Obținem frecvența notelor
        freq = dict(map(lambda x: (x, note_muzicale.count(x)), unique_notes))

        # Cream un dicționar cu indexul notei ca cheie și nota ca valoare și invers
        indnote = dict(enumerate(freq))
        noteind = dict(map(reversed, indnote.items()))

        timesteps = 114
        x = []
        y = []

        for i in notes_array:
            for j in range(0, len(i) - timesteps):
                inp = i[j:j + timesteps]
                out = i[j + timesteps]

                x.append(list(map(lambda x: noteind[x], inp)))
                y.append(noteind[out])

        xnew = np.array(x)
        ynew = np.array(y)

        xnew = np.reshape(xnew, (len(xnew), timesteps, 1))
        ynew = np.reshape(ynew, (-1, 1))

        # Încărcăm modelul salvat
        model = load_model("ModeL")
        print("Model loaded successfully.")
        index = np.random.randint(0, len(xnew) - 1)
        pattern = xnew[index]
        predicted_notes = []

        # Generăm 150 de note
        for i in range(150):
            pattern = pattern.reshape(1, len(pattern), 1)
            predicted_index = np.argmax(model.predict(pattern))
            predicted_notes.append(indnote[predicted_index])
            pattern = np.append(pattern, predicted_index)
            pattern = pattern[1:]

        outputnotes = []

        # Dacă modelul este o instanță de acord
        for offset, patt in enumerate(predicted_notes):
            if ('.' in patt) or patt.isdigit():
                # Împărțim notele din acord
                notes_in_chord = patt.split('.')
                notes = []

                for currentnote in notes_in_chord:
                    i_note = int(currentnote)
                    new_note = note.Note(i_note)
                    new_note.instrument = instrument.Piano()
                    notes.append(new_note)

                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                outputnotes.append(new_chord)

            else:
                new_note = note.Note(patt)
                new_note.offset = offset
                new_note.instrument = instrument.Piano()
                outputnotes.append(new_note)

    else:
        # Antrenarea modelului

        # Note unice
        note_muzicale = sum(notes_array, [])
        unique_notes = list(set(note_muzicale))

        # Obținem frecvența notelor
        freq = dict(map(lambda x: (x, note_muzicale.count(x)), unique_notes))

        # Cream un dicționar cu indexul notei ca cheie și nota ca valoare și invers
        indnote = dict(enumerate(freq))
        noteind = dict(map(reversed, indnote.items()))

        timesteps = 114
        x = []
        y = []

        for i in notes_array:
            for j in range(0, len(i) - timesteps):
                inp = i[j:j + timesteps]
                out = i[j + timesteps]

                x.append(list(map(lambda x: noteind[x], inp)))
                y.append(noteind[out])

        xnew = np.array(x)
        ynew = np.array(y)

        xnew = np.reshape(xnew, (len(xnew), timesteps, 1))
        ynew = np.reshape(ynew, (-1, 1))

        # Divizăm datele în seturi de antrenament și testare
        xtrain, xtest, ytrain, ytest = train_test_split(xnew, ynew, test_size=0.3, random_state=42)

        # Cream și antrenăm modelul
        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=(xnew.shape[1], xnew.shape[2])))
        model.add(Dropout(0.5))
        model.add(LSTM(256))
        model.add(Dropout(0.5))
        model.add(Dense(len(noteind), activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(xtrain, ytrain, batch_size=128, epochs=15, validation_data=(xtest, ytest))

        # Salvăm modelul antrenat
        model.save("ModeL")
        print("Model trained and saved successfully.")

        index = np.random.randint(0, len(xtest) - 1)
        pattern = xtest[index]
        predicted_notes = []

        # Generăm 150 de note
        for i in range(150):
            pattern = pattern.reshape(1, len(pattern), 1)
            predicted_index = np.argmax(model.predict(pattern))
            predicted_notes.append(indnote[predicted_index])
            pattern = np.append(pattern, predicted_index)
            pattern = pattern[1:]

        outputnotes = []

        # Dacă modelul este o instanță de acord
        for offset, patt in enumerate(predicted_notes):
            if ('.' in patt) or patt.isdigit():
                # Împărțim notele din acord
                notes_in_chord = patt.split('.')
                notes = []

                for currentnote in notes_in_chord:
                    i_note = int(currentnote)
                    new_note = note.Note(i_note)
                    new_note.instrument = instrument.Piano()
                    notes.append(new_note)

                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                outputnotes.append(new_chord)

            else:
                new_note = note.Note(patt)
                new_note.offset = offset
                new_note.instrument = instrument.Piano()
                outputnotes.append(new_note)

    # Salvăm fișierul MIDI generat
    midi_stream = stream.Stream(outputnotes)
    midi_stream.write('midi', fp='music.mid')


#interfata proiectului
window = tk.Tk()
window.title("Automatic music generation")
window.geometry("1920x1080")
window.configure(background='#c44b4b')
window.resizable(True, True)

label_font = font.Font(size=40, family='Courier New', weight='bold')
header = tk.Label(window, text="Automatic music generation", font=label_font, bg='#F4CE74', fg='#620808', padx=1000)
header.pack()

label = Label(window, text="Please enter your choice! Write the last name of "
                           "the composer in lowercase letters!", font=("Courier 22 bold"), bg='#f5f4ae')
label.pack(pady=15)

entry = Entry(window, width=20, font=("Courier 22 bold", 20))
entry.focus_set()
entry.pack(pady=5)

#tk.Button(window, text="Send", width=5, command=display_text, font=("Courier 22 bold", 15), bg='#f5f4ae').pack()
tk.Button(window, text="Generate Music", width=15, command=generate_music, font=("Courier 22 bold", 15), bg='#f5f4ae').pack()

img1 = Image.open("mozart.png")
img2 = Image.open("Beethoven.png")
img3 = Image.open("liszt.png")
img4 = Image.open("schubert.png")

resized_image= img1.resize((250,350))
Img1 = ImageTk.PhotoImage(resized_image)

resized_image= img2.resize((250,350))
Img2 = ImageTk.PhotoImage(resized_image)

resized_image= img3.resize((250,350))
Img3 = ImageTk.PhotoImage(resized_image)

resized_image= img4.resize((250,350))
Img4 = ImageTk.PhotoImage(resized_image)

mozart = tk.Label(window, image = Img1)
beethoven = tk.Label(window, image = Img2)
liszt = tk.Label(window, image = Img3)
schubert = tk.Label(window, image = Img4)
mozart.pack(padx = 62, side = tk.LEFT, fill = "none", expand = "no")
beethoven.pack(padx = 62, side = tk.LEFT, fill = "none", expand = "no")
liszt.pack(padx = 62, side = tk.LEFT, fill = "none", expand = "no")
schubert.pack(padx = 62, side = tk.LEFT, fill = "none", expand = "no")

myfont = font.Font(family = 'Georgia', size = 20)

mtext = tk.Label(window,
                 text = "Wolfgang Mozart",
                 bg = "#FEFFC2",
)

mtext['font'] = myfont
mtext.place( x = 88, y = 735 )

betext = tk.Label(window,
                  text = "Ludwig van Beethoven",
                  bg = "#FEFFC2",
)

betext['font'] = myfont
betext.place( x = 427, y = 735 )

ltext = tk.Label(window,
                  text = "Franz Liszt",
                  bg = "#FEFFC2",
)

ltext['font'] = myfont
ltext.place( x = 875, y = 735 )

stext = tk.Label(window,
                  text = "Franz Schubert",
                  bg = "#FEFFC2",
)

stext['font'] = myfont
stext.place( x = 1230, y = 735 )

window.mainloop()

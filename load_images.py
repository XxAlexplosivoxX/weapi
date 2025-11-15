import face_recognition

#el_papu_face_image = face_recognition.load_image_file("photos/yo.jpeg")
#el_papu_face_encoding = face_recognition.face_encodings(el_papu_face_image)[0]

rubiuh_face_image = face_recognition.load_image_file("photos/rubiuh.jpg")
rubiuh_face_encoding = face_recognition.face_encodings(rubiuh_face_image)[0]

known_face_encodings = [
#    el_papu_face_encoding,
    rubiuh_face_encoding
]
known_face_names = [
#    "Alex",
    "El Rubius"
]
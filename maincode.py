import cv2
import face_recognition
import os
import gc

# reduce the image size to 320*240 to save memory
img_width = 320
img_height = 240

# load the faces
imgs = []
names = []
face_dir = "all_faces_to_compare"
for file_name in os.listdir(face_dir):
    img = face_recognition.load_image_file(os.path.join(face_dir, file_name))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, (img_width, img_height))
    img_encoding = face_recognition.face_encodings(rgb_img)
    if len(img_encoding) > 0:
        img_encoding = img_encoding[0]
        imgs.append(img_encoding)
        names.append(os.path.splitext(file_name)[0])

# initialize the camera
#cap = cv2.VideoCapture("http://192.168.1.168:81/stream")
video_capture = cv2.VideoCapture('http://192.168.1.168:81/stream')
#video_capture.set(cv2.CAP_PROP_FPS, 120)

# reduce the frame size to 320*240 to save memory
#video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)

while True:
    # read a frame from the camera
    ret, frame = video_capture.read()
    if not ret:
        break

    # convert the frame to RGB and resize it
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (img_width, img_height))

    # find the face locations and encodings in resize it
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # compare the face encodings to the img in folder
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(imgs, face_encoding)
        name = "Unkown"

        if True in matches:
            first_match_index = matches.index(True)
            name = names[first_match_index]

        # draw a rectangle around the face and display the name
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    # display the frame
    cv2.imshow('video', frame)

    # Release the memory of the resized images and garbage collect to free memory
    del resized_frame, face_locations, face_encodings
    gc.collect()

    # exit on 's' key press
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# release the camera and destroy the windows
video_capture.release()
cv2.destroyAllWindows()

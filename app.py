"""
    Module contains a simple UI to test our detector using existing video or with
    real-time video from the webcam.
        
    @author  Mohamed Hassan
    @since   2024-5-3
"""

import tkinter as tk
from tkinter import filedialog
from detector import webcam_handler, video_handler


def browse_file():
    """
    Helper function that called when the 'Browse' button clicked.
    It graps the path of the chosen video and pass it to the video_handler() function.

    @param: None
    @return: None
    """

    filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    video_handler(filename)


root = tk.Tk()
root.title("Video Selection")

webcam_button = tk.Button(
    root, text="Webcam", command=webcam_handler, height=20, width=80, bg="light green"
)
webcam_button.pack(pady=10)

browse_button = tk.Button(
    root, text="Browse", command=browse_file, height=20, width=80, bg="light blue"
)
browse_button.pack(pady=10)

root.mainloop()

# Real-time Object Detector

This is the first task in the [CodeAlpha](https://www.codealpha.tech/) AI internship.

This repository contains code for a real-time object tracker and detection system implemented using the **YOLO** (You Only Look Once) model specifically, **YOLOv8**. The system is capable of detecting and tracking objects in existing videos as well as in real-time using a webcam feed.


## Overview

The project is divided into the following files:

- `detector.py`: module contains the work with the YOLO model in the two scenarios.
- `app.py`: module contains simple UI using `tkinter` that deals with `detector.py` in the background.
- `requirements.txt`: dependencies needed to be able to run the project.


## Tools

- [**YOLOv8**](https://docs.ultralytics.com/).
- **OpenCV**.


## Features

- Object detection and tracking in existing videos.
- Real-time object detection using a webcam.
- Support for multiple object classes.
- Adjustable versions of the **YOLO** model.


## Demo

![Demo](Demo.gif)


## Usage

1. Clone the repository:

    ```bash
    git clone git@github.com:mohamedhassan218/CodeAlpha_Object_Detector.git
    ```

2. Create a Virtual Environment:
    ```bash
    python -m venv .venv
    ```

3. Activate the Virtual Environment:
    - On Windows:
        ```bash
        .venv\Scripts\activate
        ```

    - On Unix or MacOS:
        ```bash
        source .venv/bin/activate
        ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Run the main script:

    ```bash
    py app.py
    ```


## Contributing

Contributions are welcome! If you have any suggestions, bug fixes, or improvements, please feel free to open an issue or create a pull request.
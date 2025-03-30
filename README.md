README.md


General Info:

    This project implements an edge detection system using both traditional computer vision techniques and deep learning (U-Net) on the BSDS500 dataset. The goal is to compare the performance of traditional and modern approaches for detecting edges in grayscale images.


Approach/Model Selection:

    My main tool was Tensorflow. I had already used this library for NLP projects, so it was not difficult to adapt it for 
    edge detection. 

    I started with just traditional edge-detection using canny and sobel. I started small, and built up slowly.

    A few challanges I encountered involved file paths, so I had to write in some error-checking. This is seperate from the
    obvious learning curve, which was not too hard to surpass.

Result/Analysis:

Method	Pros	                        Cons

Canny	Fast and sharp edge results	    Sensitive to noise and thresholds

Sobel	Simple gradient-based edges	    Detects thick and fuzzy edges

U-Net	Learns contextual edge patterns	Slower training; needs more data for best performance

Final Accuracy ~95%

Loss trend: Decreased from ~0.61 to ~0.11 over 10 epochs
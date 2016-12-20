# CameraPoseEstimation

This is a small program to estimate the single camera position and orientation when given pictures of a pattern.

Click here [Demo YouTube Video](https://youtu.be/yDYZd8huJh4) to watch a demo video!

### Author
Zongchang (Jim) Chen - Junior student at Haverford College, class of 2018.

Computer Science & Mathematics.

## Dependencies
To run this program, you'll have to install

* ```Python 2.7.x``` and the following modules: ```numpy```, ```scipy```, ```matplotlib```
* ```OpenCV 2.x``` My version is ```2.4.11```

On Linux, you can type this line in Terminal to install all ```Python``` modules:

```$ sudo apt-get install python-dev python-numpy python-matplotlib python-scipy```

On Mac OSX, you can do:

```$ pip install numpy matplotlib scipy```

Or I highly recommend to install [```Anaconda```](https://www.continuum.io/downloads) which would automatically install all necessary ```Python``` libraries on your local computer.

My version of ```Anaconda``` is ```4.2.9```. More information for ```Anaconda```, please check [here](https://www.continuum.io/downloads).

Another huge advantage of using ```Anaconda``` is its simplification of installing ```Python OpenCV```. 
You may find painful to download ```OpenCV``` source code and compile by yourself.

By ```Anaconda```, you could just type this line in your Terminal:

```conda install opencv```

If this does not work, check more on this [thread](http://stackoverflow.com/questions/23119413/how-to-install-python-opencv-through-conda)

## How to use
Before run, make sure that you have the pattern image in folder ```pattern``` and images in folder ```images```.

But you can actually modify the directories in file ```src/settings.py```. The paths mentioned above are just set by default.

Everything is already setup when you download all files from this repository, you can just type ```python src/main.py``` on your local Terminal and then play with the model shown up.

If you want to add some more pictures, you can add them in the folder ```images``` and then modify the code block in ```src/main.py```.

## Some notices
This program is coded in Mac OSX environment, so Mac users do not need to do any modifications on the code. 

But since ```OpenCV``` on Mac have some issue on function ```cv2.imread```, it rotates the input image by 90 degree counter-clockwise sometimes when the image is too huge. 
So there is one line to rotate it back in the code. If you are not using Mac, try to run the code at first, and if it does not work well, delete Line 37 in file ```camera_pose_estimate.py``` and try it again (sorry for Windows users （＞人＜；）).

Enjoy :)

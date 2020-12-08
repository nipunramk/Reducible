# Reducible
Repository containing all code for the videos on the Reducible YouTube channel.
This entire repository makes use of 3blue1brown's open source library manim: https://github.com/3b1b/manim

In general, I avoided changing too much of the source of manim to minimize the issues others may have trying out the code. There is one case where I changed the source code for manim for the graph theory videos. If you run into any issues with running the graph theory videos, see this issue: https://github.com/nipunramk/Reducible/issues/1. Another case where I have modified the source library for manim is adding colors in the manimlib/constants.py. If you see colors not defined when trying to run the code, you must define the hex value of the color inside the COLOR_MAP dictionary in that file. 

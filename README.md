# Reducible
Repository containing all code for the videos on the Reducible YouTube channel.
This entire repository makes use of 3blue1brown's open source library manim: https://github.com/3b1b/manim 
In the more recent videos, I have used Manim Community edition: https://www.manim.community/

### Some updates to the Repo as of October 31, 2021
On the latest video relating to marching squares (2021/MarchingSquares/scene.py, I have decided to start using the Manim Community version. The version has been well-maintained by a group of developers and more documentation on setup of Manim Community can be found here: https://www.manim.community/
All other videos utilize an old version of the manim repo from https://github.com/3b1b/manim

In general, I avoided changing too much of the source of manim to minimize the issues others may have trying out the code. There is one case where I changed the source code for manim for the graph theory videos. If you run into any issues with running the graph theory videos, see this issue: https://github.com/nipunramk/Reducible/issues/1. Another case where I have modified the source library for manim is adding colors in the manimlib/constants.py. If you see colors not defined when trying to run the code, you must define the hex value of the color inside the COLOR_MAP dictionary in that file. 

### Some notes on changes to repository as of March 23, 2021
For the 2021/GJK/gjk.py code, which contains all the animations for the GJK video, one new dependency is required in addition to manim dependencies: the Python Shapely library (used for some animations involving concave shape intersection and union -- a problem I tried hacking on but ran into too many issues to be able to use in a video -- this library saved the day).

Installation instructions for Shapely can be found here: https://pypi.org/project/Shapely/
Documentation can be found here: https://shapely.readthedocs.io/en/stable/manual.html

Additionally, I have updated the manimlib from the original 2019 version I was using for my first few videos to the more recent May 2020 version that works quite well for the more recent videos. I have made some minor additions/modifications to the library for my own use cases as well. Note, this version of manim still uses Cairo as a backend. There are several versions of manim now, the one I recommend for most people is the Manim Community Edition (CE) which has good documentation and good community support. I started this channel before this existed so I spent a lot of time digging into the library on my own, but now, with Manim CE, development and support is significantly easier.

There is also a ManimGL version that uses openGL https://github.com/3b1b/manim. I am still figuring out this version of manim and I may move towards it in the future for some projects I have in mind. I do not recommend starting with the openGL version, since it is still not quite stable and still a work in progress on Grant's side of things as far as I know. It is also much trickier to learn, but that's just my opinion. 

# Hackathon-3-27-2025
Ever since the pandemic a few years ago, people have been exercising more and more inside their own homes. However, many people do not know how to properly do exercise, or they do not have a way to check if they are doing the exercise correctly. To remedy this, I created a program that runs in real time with acceptable accuracy to help assist people in performing squats with the correct form. 

This program uses a pre trained model, which is Tensorflow’s Movenet Lightning. Movenet has several different versions, all of which are viable options that I could have used. I used lightning because it has the lowest latency, which is needed for the program I am using it for.

The program counts the number of squats you have completed with correct form, giving you one bell sound when you are in squat position, and two bell sounds when you stand up. It shows your current right knee angle and left knee angle in the terminal, as well as your current squat count.


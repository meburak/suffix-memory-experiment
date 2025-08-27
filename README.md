### Free Recall Memory Experiment with Suffixes 
PsychoPy based memory experiment, with which you can: 
- Add and remove lists, 
- Add and remove suffixes, 
- Change routine times, 
- Change list lengths, 
- Get the file output with recall and presentation positions, 
- Get the intrusions. 
Project is working but still needs some work and touch-ups. 
There is a seperate python script provided to clean and edit the data. 
Built with PsychoPy v2025.1.1
***
#### How to use the experiment file
You will need PsychoPy v2025.1.1 to run and edit the experiment. It is recomended to run the experiment once before doing changes.
The file is currently configurated for a free recall type of experiment, however it can be made into a serial recall experiment. 
It is necessary, to run the experiment at least once and having the column letters for editing the "config.ini" file. Also, make sure before proceeding to data collection the experiment works like it is supposed to and the participants are comfortable with the process. Always check the output data prior to collection. 
***
#### Changing the stimuli lists, words 
To change the words, use "word_list_main.xlsx" and modify the "Words" column. 
After adding, removing or changing the words there go into experiment builder. 
There, find "study_" routines. Adjust the input gaps, accordingly. 
In "recall_" routines, find the related loops, change the loop count to match the item count for related study period. 
***
#### Time adjustments
Changing the recall periods time limit: 
Before the related routine, there is a clock_start routine.
In that, go to code_clock_xxx,
There change the time limit to whatever in seconds, default is 120. 
***
#### Sound 
For proper execution of the sound files they should be formatted .wav and they should have 48kHz frequency. Audacity can be used to format the files.  
***
#### Routines and loops
consent - a screen for displaying consent message. if not agreed experiment  terminates.
tutorial - routine for displaying experiment instructions.
dep - a screen to get department input. can be used for other demographics.
if needed can be multiplied.
trial_Word_Display - study words get showcased.
clock_start_[xxx] - for measuring time. used to terminate the recall routine.  adjust time, read more on edit section
recall_[xxx] - a recall period for the list [xxx]
study_[xxx] - a study session where items are shown randomly.
proceed_bridge - a link between previous list ending and beginning of the next  list. Used for letting participant know that next sequence is on the way.
suffix_play - here the suffix sounds are played. can be removed if a pure free  recall experiment is desired
***
#### Script
The experiment should automatically run the script "clean.py", it is used to format and clean the data into neat .csv files. It also outputs the positions as well.The cleaned data can be found on cleanedData folder. 
To edit values you can use "config.ini", as of 27-08-2025 the numTolerate and tiralyes values are not used. I am planning to use them later on. Important to edit are: 
- numLists = number of lists except tutorial list. 
- columnsExtract = columns that are going to be extracted. It handles the text inputs for recall part, words and reaction times. 
			

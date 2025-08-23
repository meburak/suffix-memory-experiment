***************
The project was designed to study a phenomenon called Suffix Effect, with Free Recall. 
 
However it can be modified for serial recall, and other types. 

by -Burak

For contact: bsyclecs@gmail.com

***************
Built with PsychoPy v2025.1.1

No license, I guess, I do not really know about those. 
Use freely. 

Current version uses a separate Python script for sorting and cleaning data.
You can find it titled as "clean.py"

It tolerates up to one mistake in form of either of the following:
deletion, substitution, addition. You can change that in script.

NOTES: 
- In case the audio files have issues with playback, use 
Audacity to change frequency settings to 48 kHz. Can find how to on Google.
Took me 2 straight days of work to fix it, I hope you do not have to. 

***************
CHANGING WORDS DISPLAYED: (more detail on Edit section)  		
***************
	Go into "word_list_main.xlsx"
	Put in your words into rows. 
	After column name start with 0. For example, for column "Words" it works like defining a list: Words = [T1,..]
	So Words[0] is T1. 
	The x is included and y is not: [x, y) 	

		Trial: 0 - 6 -> so we have 0, 1, 2, 3, 4, 5 
		List 1: 6 - 24 
		List 2: 24 - 42
		List 3: 42 - 60 
		List 4: 60 - 78 

	Of course all of these values are modular and can be adjusted to fit
	different length of stimuli lists. 

***************
ELEMENTS:
***************

	consent - a screen for displaying consent message. if not agreed experiment 	terminates. 

	tutorial - routine for displaying experiment instructions. 

	dep - a screen to get department input. can be used for other demographics. 
	if needed can be multiplied. 
	
	trial_Word_Display - study words get showcased. 

	clock_start_[xxx] - for measuring time. used to terminate the recall routine. 	adjust time, read more on edit section

	recall_[xxx] - a recall period for the list [xxx]

	study_[xxx] - a study session where items are shown randomly. 

	proceed_bridge - a link between previous list ending and beginning of the next 	list. Used for letting participant know that next sequence is on the way. 

	suffix_play - here the suffix sounds are played. can be removed if a pure free 	recall experiment is desired. 

	

**************
HOW TO EDIT THE EXPERIMENT
**************

	Adding / Removing Lists: 

		The procedure to be followed here goes like this: 
		For normal (non-suffix) lists we have four parts or routines.
		
		-Study, seen as study_[xxx]
			This part is where the study section is, where the words are 
			displayed, one by one. You can adjust the words, source file, 
			randomization etc. from this section 

		-clock_start_[xxx]
			Here, a local clock is initiated to limit the exposure of recall 
			routine. Time can be adjusted in here from code components' 
			time_limit section. 
			
		-recall_[xxx]
			A recall routine. Mouse 1 skips, ENTER saves the response. 
		
		-proceed_bridge[xxx]
			Not necessary but good to have. 

		To add a list you should: 
			Copy the above discussed routines, and rename accordingly. The 				sequence should not change. Clock should always be put RIGHT BEFORE
			the recall. 

		To remove, just click Mouse 2 on routine, and click remove. 

	Changing Items, Item Counts, Randomization etc: 
		
		-For each study routine go to related section of loops; 
		e.g list4_words, then there: 
		
		-Conditions: put in an excel file or csv to pull the items. This must be 		the show the items. You can put in separate files for each routine, in this 		case there is only one file for every single routine. 

		-Should for each routine, in text_[x], $ColumnName be put. ColumnName is 
		for the column that contains the items you want to showcase. 
		For the default version, I used $Words because my items were saved on
		that column. 

		-Selected Rows: here select the rows where the related sections are.
		e.g for list_4, I am selecting corresponding items: 62:79 (18 items)

		-Loop Type: random 
		is for making the items showcased random, each item gets displayed once. 

	Changing the Displayed Texts: 
		
		Go to related routine, and find the text_[***] 
		Fill the box called "Text" with whatever you want. 

	Changing Time given in Recall Period: 
		
		Before the related routine, there is a clock_start routine. 
		In that, go to code_clock_xxx, 
		There change the time limit to whatever in seconds, default is 120.

	Sound Stimuli (Suffix): 
	
		The sound files provided as laminer48.wav, and aktuator48.wav are used in 
		suffix_play routines. You can change them, by clicking the routine
		and chosing the sound file you want. 
		NOTE: should be .wav file and in 48khz format. You can use Audacity to
		change the wavelengths. 

**************
DATA
**************
Collected in folder "data". It is in raw format and looks terrible, the python script provided fixes and clears the data.

It uses a Levenshtein Distance algorithm, which accounts for mistakes when recall happens. 
More information is provided in "clean.py".


		
			
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on August 17, 2025, at 18:08
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'memoryexp_recall_word_input'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\elito\\OneDrive\\Desktop\\Keep\\Code&Assets\\psychopy-recall-suffix\\suffix_experiment - stable_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('data')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('consent_resp') is None:
        # initialise consent_resp
        consent_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='consent_resp',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_recall_Trial') is None:
        # initialise key_resp_recall_Trial
        key_resp_recall_Trial = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_recall_Trial',
        )
    if deviceManager.getDevice('key_resp_list_1_recall') is None:
        # initialise key_resp_list_1_recall
        key_resp_list_1_recall = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_list_1_recall',
        )
    # create speaker 'sound_1'
    deviceManager.addDevice(
        deviceName='sound_1',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1,
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_list_1_recall_2') is None:
        # initialise key_resp_list_1_recall_2
        key_resp_list_1_recall_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_list_1_recall_2',
        )
    if deviceManager.getDevice('key_resp_list_1_recall_3') is None:
        # initialise key_resp_list_1_recall_3
        key_resp_list_1_recall_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_list_1_recall_3',
        )
    # create speaker 'sound_2'
    deviceManager.addDevice(
        deviceName='sound_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1,
        resample='True',
        latencyClass=1,
    )
    if deviceManager.getDevice('key_resp_list_1_recall_4') is None:
        # initialise key_resp_list_1_recall_4
        key_resp_list_1_recall_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_list_1_recall_4',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "consent" ---
    consent_resp = keyboard.Keyboard(deviceName='consent_resp')
    consent_text = visual.TextStim(win=win, name='consent_text',
        text='This experiment is designed to observe the "Suffix Effect,"\na phenomenon related to memory.\n\nIt is not a performance assessment. You may exit the study at any time without providing a reason.\n\nTo continue, press the "Y" key.\nTo quit, press the "N" key.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from code_consent
    from psychopy import prefs
    prefs.hardware['audioLib'] = ['sounddevice', 'pyo', 'pygame']
    
    
    
    # --- Initialize components for Routine "tutorial" ---
    text_Tutorial = visual.TextStim(win=win, name='text_Tutorial',
        text='Our experiment consists of two repeating sections.\n\nIn the first section, a series of words will appear sequentially on the screen. You will be asked to memorize these words as they are displayed.\n\nIn the second phase, you must type these words on the screen and press the "ENTER" key. After typing each word you recall, press "ENTER." You may begin typing once the screen turns green. You will have two minutes for this section, but you can skip ahead at any time by pressing the "LEFT MOUSE" button.\n\nYou will repeat these two sections four times in total. Before the main task, you will complete a short practice trial.\n\nTo proceed, press the "Y" key to advance to the practice section.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "bye_message" ---
    text_Bye = visual.TextStim(win=win, name='text_Bye',
        text='You have NOT given consent\n\nThanks for attending',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "dep" ---
    departmentInput = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         ori=0.0, pos=(0, -0.4), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='departmentInput',
         depth=0, autoLog=True,
    )
    department = visual.TextStim(win=win, name='department',
        text='Please provide your department, then proceed with "ENTER" key.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "trial_Word_Display" ---
    trial_text = visual.TextStim(win=win, name='trial_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "clock_start" ---
    # Run 'Begin Experiment' code from code_clock_trial
    total_time = 0.0  # Track cumulative time  
    time_limit = 120.0 # 40-second limit  
    global_clock = core.Clock()  # Global timer  
    
    
    # --- Initialize components for Routine "recall_Trial" ---
    textbox_recall_Trial = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_recall_Trial',
         depth=-1, autoLog=False,
    )
    key_resp_recall_Trial = keyboard.Keyboard(deviceName='key_resp_recall_Trial')
    mouse_trial = event.Mouse(win=win)
    x, y = [None, None]
    mouse_trial.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "proceed_study" ---
    bridge_Text = visual.TextStim(win=win, name='bridge_Text',
        text='YOU HAVE COMPLATED THE TRIAL PART.\n\nMoving on to the experiment.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_3 = visual.TextStim(win=win, name='text_3',
        text='LIST 1',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "study_list1" ---
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "clock_start_list_1" ---
    # Run 'Begin Experiment' code from code_clock_list_1
    total_time = 0.0  # Track cumulative time  
    time_limit = 120.0 # 40-second limit  
    global_clock = core.Clock()  # Global timer  
    
    
    # --- Initialize components for Routine "recall_list1" ---
    list1_recall = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='list1_recall',
         depth=0, autoLog=True,
    )
    key_resp_list_1_recall = keyboard.Keyboard(deviceName='key_resp_list_1_recall')
    mouse_1 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_1.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "proceed_bridge" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='LIST 2',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "study_list2" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "suffix_play" ---
    sound_1 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=False, 
        hamming=True, 
        speaker='sound_1',    name='sound_1'
    )
    sound_1.setVolume(1.0)
    
    # --- Initialize components for Routine "clock_start_list_2" ---
    # Run 'Begin Experiment' code from code_clock_list
    total_time = 0.0  # Track cumulative time  
    time_limit = 120.0 # 40-second limit  
    global_clock = core.Clock()  # Global timer  
    
    
    # --- Initialize components for Routine "recall_list2" ---
    list2_recall = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='list2_recall',
         depth=0, autoLog=True,
    )
    key_resp_list_1_recall_2 = keyboard.Keyboard(deviceName='key_resp_list_1_recall_2')
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "proceed_bridge_2" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text='LIST 3',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "study_list3" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "clock_start_list_3" ---
    # Run 'Begin Experiment' code from code_clock_list_2
    total_time = 0.0  # Track cumulative time  
    time_limit = 120.0 # 40-second limit  
    global_clock = core.Clock()  # Global timer  
    
    
    # --- Initialize components for Routine "recall_list3" ---
    list3_recall = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='list3_recall',
         depth=0, autoLog=True,
    )
    key_resp_list_1_recall_3 = keyboard.Keyboard(deviceName='key_resp_list_1_recall_3')
    mouse_3 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_3.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "proceed_birdge_3" ---
    text_7 = visual.TextStim(win=win, name='text_7',
        text='LIST 4',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "study_list4" ---
    text_8 = visual.TextStim(win=win, name='text_8',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "clock_start_list_4" ---
    # Run 'Begin Experiment' code from code_clock_list_3
    total_time = 0.0  # Track cumulative time  
    time_limit = 120.0 # 40-second limit  
    global_clock = core.Clock()  # Global timer  
    
    
    # --- Initialize components for Routine "suffix_play_2" ---
    sound_2 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=False, 
        hamming=True, 
        speaker='sound_2',    name='sound_2'
    )
    sound_2.setVolume(1.0)
    
    # --- Initialize components for Routine "recall_list4" ---
    list4_recall = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='list4_recall',
         depth=0, autoLog=True,
    )
    key_resp_list_1_recall_4 = keyboard.Keyboard(deviceName='key_resp_list_1_recall_4')
    mouse_4 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_4.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "end" ---
    text_end = visual.TextStim(win=win, name='text_end',
        text='THANK YOU! <3',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "consent" ---
    # create an object to store info about Routine consent
    consent = data.Routine(
        name='consent',
        components=[consent_resp, consent_text],
    )
    consent.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for consent_resp
    consent_resp.keys = []
    consent_resp.rt = []
    _consent_resp_allKeys = []
    # store start times for consent
    consent.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    consent.tStart = globalClock.getTime(format='float')
    consent.status = STARTED
    consent.maxDuration = None
    # keep track of which components have finished
    consentComponents = consent.components
    for thisComponent in consent.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "consent" ---
    consent.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *consent_resp* updates
        waitOnFlip = False
        
        # if consent_resp is starting this frame...
        if consent_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            consent_resp.frameNStart = frameN  # exact frame index
            consent_resp.tStart = t  # local t and not account for scr refresh
            consent_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(consent_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            consent_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(consent_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(consent_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if consent_resp.status == STARTED and not waitOnFlip:
            theseKeys = consent_resp.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=False)
            _consent_resp_allKeys.extend(theseKeys)
            if len(_consent_resp_allKeys):
                consent_resp.keys = _consent_resp_allKeys[-1].name  # just the last key pressed
                consent_resp.rt = _consent_resp_allKeys[-1].rt
                consent_resp.duration = _consent_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *consent_text* updates
        
        # if consent_text is starting this frame...
        if consent_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            consent_text.frameNStart = frameN  # exact frame index
            consent_text.tStart = t  # local t and not account for scr refresh
            consent_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(consent_text, 'tStartRefresh')  # time at next scr refresh
            # update status
            consent_text.status = STARTED
            consent_text.setAutoDraw(True)
        
        # if consent_text is active this frame...
        if consent_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=consent,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            consent.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in consent.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "consent" ---
    for thisComponent in consent.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for consent
    consent.tStop = globalClock.getTime(format='float')
    consent.tStopRefresh = tThisFlipGlobal
    # check responses
    if consent_resp.keys in ['', [], None]:  # No response was made
        consent_resp.keys = None
    thisExp.addData('consent_resp.keys',consent_resp.keys)
    if consent_resp.keys != None:  # we had a response
        thisExp.addData('consent_resp.rt', consent_resp.rt)
        thisExp.addData('consent_resp.duration', consent_resp.duration)
    # Run 'End Routine' code from code_consent
    if consent_resp.keys == "y": 
        consent_given = 1
        consent_not_given = 0 
    elif consent_resp.keys == "n": 
        consent_given = 0
        consent_not_given = 1
    
    event.clearEvents(eventType='keyboard')
    thisExp.nextEntry()
    # the Routine "consent" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    consent_Tutorial = data.TrialHandler2(
        name='consent_Tutorial',
        nReps=consent_given, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(consent_Tutorial)  # add the loop to the experiment
    thisConsent_Tutorial = consent_Tutorial.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisConsent_Tutorial.rgb)
    if thisConsent_Tutorial != None:
        for paramName in thisConsent_Tutorial:
            globals()[paramName] = thisConsent_Tutorial[paramName]
    
    for thisConsent_Tutorial in consent_Tutorial:
        consent_Tutorial.status = STARTED
        if hasattr(thisConsent_Tutorial, 'status'):
            thisConsent_Tutorial.status = STARTED
        currentLoop = consent_Tutorial
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisConsent_Tutorial.rgb)
        if thisConsent_Tutorial != None:
            for paramName in thisConsent_Tutorial:
                globals()[paramName] = thisConsent_Tutorial[paramName]
        
        # set up handler to look after randomisation of conditions etc
        tutorial_loop = data.TrialHandler2(
            name='tutorial_loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(tutorial_loop)  # add the loop to the experiment
        thisTutorial_loop = tutorial_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTutorial_loop.rgb)
        if thisTutorial_loop != None:
            for paramName in thisTutorial_loop:
                globals()[paramName] = thisTutorial_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTutorial_loop in tutorial_loop:
            tutorial_loop.status = STARTED
            if hasattr(thisTutorial_loop, 'status'):
                thisTutorial_loop.status = STARTED
            currentLoop = tutorial_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTutorial_loop.rgb)
            if thisTutorial_loop != None:
                for paramName in thisTutorial_loop:
                    globals()[paramName] = thisTutorial_loop[paramName]
            
            # --- Prepare to start Routine "tutorial" ---
            # create an object to store info about Routine tutorial
            tutorial = data.Routine(
                name='tutorial',
                components=[text_Tutorial, key_resp],
            )
            tutorial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for key_resp
            key_resp.keys = []
            key_resp.rt = []
            _key_resp_allKeys = []
            # store start times for tutorial
            tutorial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            tutorial.tStart = globalClock.getTime(format='float')
            tutorial.status = STARTED
            tutorial.maxDuration = None
            # keep track of which components have finished
            tutorialComponents = tutorial.components
            for thisComponent in tutorial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "tutorial" ---
            tutorial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisTutorial_loop, 'status') and thisTutorial_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_Tutorial* updates
                
                # if text_Tutorial is starting this frame...
                if text_Tutorial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_Tutorial.frameNStart = frameN  # exact frame index
                    text_Tutorial.tStart = t  # local t and not account for scr refresh
                    text_Tutorial.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_Tutorial, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_Tutorial.status = STARTED
                    text_Tutorial.setAutoDraw(True)
                
                # if text_Tutorial is active this frame...
                if text_Tutorial.status == STARTED:
                    # update params
                    pass
                
                # *key_resp* updates
                waitOnFlip = False
                
                # if key_resp is starting this frame...
                if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp.frameNStart = frameN  # exact frame index
                    key_resp.tStart = t  # local t and not account for scr refresh
                    key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp.getKeys(keyList=['y'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_allKeys.extend(theseKeys)
                    if len(_key_resp_allKeys):
                        key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                        key_resp.rt = _key_resp_allKeys[-1].rt
                        key_resp.duration = _key_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=tutorial,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    tutorial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in tutorial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "tutorial" ---
            for thisComponent in tutorial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for tutorial
            tutorial.tStop = globalClock.getTime(format='float')
            tutorial.tStopRefresh = tThisFlipGlobal
            # check responses
            if key_resp.keys in ['', [], None]:  # No response was made
                key_resp.keys = None
            tutorial_loop.addData('key_resp.keys',key_resp.keys)
            if key_resp.keys != None:  # we had a response
                tutorial_loop.addData('key_resp.rt', key_resp.rt)
                tutorial_loop.addData('key_resp.duration', key_resp.duration)
            # Run 'End Routine' code from code
            event.clearEvents(eventType='keyboard')
            # the Routine "tutorial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisTutorial_loop as finished
            if hasattr(thisTutorial_loop, 'status'):
                thisTutorial_loop.status = FINISHED
            # if awaiting a pause, pause now
            if tutorial_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                tutorial_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'tutorial_loop'
        tutorial_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisConsent_Tutorial as finished
        if hasattr(thisConsent_Tutorial, 'status'):
            thisConsent_Tutorial.status = FINISHED
        # if awaiting a pause, pause now
        if consent_Tutorial.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            consent_Tutorial.status = STARTED
    # completed consent_given repeats of 'consent_Tutorial'
    consent_Tutorial.status = FINISHED
    
    
    # set up handler to look after randomisation of conditions etc
    consent_Bye = data.TrialHandler2(
        name='consent_Bye',
        nReps=consent_not_given, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(consent_Bye)  # add the loop to the experiment
    thisConsent_Bye = consent_Bye.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisConsent_Bye.rgb)
    if thisConsent_Bye != None:
        for paramName in thisConsent_Bye:
            globals()[paramName] = thisConsent_Bye[paramName]
    
    for thisConsent_Bye in consent_Bye:
        consent_Bye.status = STARTED
        if hasattr(thisConsent_Bye, 'status'):
            thisConsent_Bye.status = STARTED
        currentLoop = consent_Bye
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisConsent_Bye.rgb)
        if thisConsent_Bye != None:
            for paramName in thisConsent_Bye:
                globals()[paramName] = thisConsent_Bye[paramName]
        
        # set up handler to look after randomisation of conditions etc
        bye_loop = data.TrialHandler2(
            name='bye_loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(bye_loop)  # add the loop to the experiment
        thisBye_loop = bye_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisBye_loop.rgb)
        if thisBye_loop != None:
            for paramName in thisBye_loop:
                globals()[paramName] = thisBye_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisBye_loop in bye_loop:
            bye_loop.status = STARTED
            if hasattr(thisBye_loop, 'status'):
                thisBye_loop.status = STARTED
            currentLoop = bye_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisBye_loop.rgb)
            if thisBye_loop != None:
                for paramName in thisBye_loop:
                    globals()[paramName] = thisBye_loop[paramName]
            
            # --- Prepare to start Routine "bye_message" ---
            # create an object to store info about Routine bye_message
            bye_message = data.Routine(
                name='bye_message',
                components=[text_Bye],
            )
            bye_message.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for bye_message
            bye_message.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            bye_message.tStart = globalClock.getTime(format='float')
            bye_message.status = STARTED
            bye_message.maxDuration = None
            # keep track of which components have finished
            bye_messageComponents = bye_message.components
            for thisComponent in bye_message.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "bye_message" ---
            bye_message.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 5.0:
                # if trial has changed, end Routine now
                if hasattr(thisBye_loop, 'status') and thisBye_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_Bye* updates
                
                # if text_Bye is starting this frame...
                if text_Bye.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_Bye.frameNStart = frameN  # exact frame index
                    text_Bye.tStart = t  # local t and not account for scr refresh
                    text_Bye.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_Bye, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_Bye.status = STARTED
                    text_Bye.setAutoDraw(True)
                
                # if text_Bye is active this frame...
                if text_Bye.status == STARTED:
                    # update params
                    pass
                
                # if text_Bye is stopping this frame...
                if text_Bye.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_Bye.tStartRefresh + 5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_Bye.tStop = t  # not accounting for scr refresh
                        text_Bye.tStopRefresh = tThisFlipGlobal  # on global time
                        text_Bye.frameNStop = frameN  # exact frame index
                        # update status
                        text_Bye.status = FINISHED
                        text_Bye.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=bye_message,
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    bye_message.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in bye_message.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "bye_message" ---
            for thisComponent in bye_message.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for bye_message
            bye_message.tStop = globalClock.getTime(format='float')
            bye_message.tStopRefresh = tThisFlipGlobal
            # Run 'End Routine' code from code_terminate
            quit(thisExp)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if bye_message.maxDurationReached:
                routineTimer.addTime(-bye_message.maxDuration)
            elif bye_message.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-5.000000)
            # mark thisBye_loop as finished
            if hasattr(thisBye_loop, 'status'):
                thisBye_loop.status = FINISHED
            # if awaiting a pause, pause now
            if bye_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                bye_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'bye_loop'
        bye_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # mark thisConsent_Bye as finished
        if hasattr(thisConsent_Bye, 'status'):
            thisConsent_Bye.status = FINISHED
        # if awaiting a pause, pause now
        if consent_Bye.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            consent_Bye.status = STARTED
    # completed consent_not_given repeats of 'consent_Bye'
    consent_Bye.status = FINISHED
    
    
    # --- Prepare to start Routine "dep" ---
    # create an object to store info about Routine dep
    dep = data.Routine(
        name='dep',
        components=[departmentInput, department, key_resp_2],
    )
    dep.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    departmentInput.reset()
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for dep
    dep.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    dep.tStart = globalClock.getTime(format='float')
    dep.status = STARTED
    thisExp.addData('dep.started', dep.tStart)
    dep.maxDuration = None
    # keep track of which components have finished
    depComponents = dep.components
    for thisComponent in dep.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "dep" ---
    dep.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *departmentInput* updates
        
        # if departmentInput is starting this frame...
        if departmentInput.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            departmentInput.frameNStart = frameN  # exact frame index
            departmentInput.tStart = t  # local t and not account for scr refresh
            departmentInput.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(departmentInput, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'departmentInput.started')
            # update status
            departmentInput.status = STARTED
            departmentInput.setAutoDraw(True)
        
        # if departmentInput is active this frame...
        if departmentInput.status == STARTED:
            # update params
            pass
        
        # *department* updates
        
        # if department is starting this frame...
        if department.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            department.frameNStart = frameN  # exact frame index
            department.tStart = t  # local t and not account for scr refresh
            department.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(department, 'tStartRefresh')  # time at next scr refresh
            # update status
            department.status = STARTED
            department.setAutoDraw(True)
        
        # if department is active this frame...
        if department.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=dep,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            dep.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in dep.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "dep" ---
    for thisComponent in dep.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for dep
    dep.tStop = globalClock.getTime(format='float')
    dep.tStopRefresh = tThisFlipGlobal
    thisExp.addData('dep.stopped', dep.tStop)
    thisExp.addData('departmentInput.text',departmentInput.text)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "dep" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_words = data.TrialHandler2(
        name='trials_words',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'word_list_main.xlsx', 
        selection='0:6'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(trials_words)  # add the loop to the experiment
    thisTrials_word = trials_words.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_word.rgb)
    if thisTrials_word != None:
        for paramName in thisTrials_word:
            globals()[paramName] = thisTrials_word[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_word in trials_words:
        trials_words.status = STARTED
        if hasattr(thisTrials_word, 'status'):
            thisTrials_word.status = STARTED
        currentLoop = trials_words
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_word.rgb)
        if thisTrials_word != None:
            for paramName in thisTrials_word:
                globals()[paramName] = thisTrials_word[paramName]
        
        # --- Prepare to start Routine "trial_Word_Display" ---
        # create an object to store info about Routine trial_Word_Display
        trial_Word_Display = data.Routine(
            name='trial_Word_Display',
            components=[trial_text],
        )
        trial_Word_Display.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        trial_text.setText(Words)
        # store start times for trial_Word_Display
        trial_Word_Display.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_Word_Display.tStart = globalClock.getTime(format='float')
        trial_Word_Display.status = STARTED
        trial_Word_Display.maxDuration = None
        # keep track of which components have finished
        trial_Word_DisplayComponents = trial_Word_Display.components
        for thisComponent in trial_Word_Display.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_Word_Display" ---
        trial_Word_Display.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # if trial has changed, end Routine now
            if hasattr(thisTrials_word, 'status') and thisTrials_word.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *trial_text* updates
            
            # if trial_text is starting this frame...
            if trial_text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                trial_text.frameNStart = frameN  # exact frame index
                trial_text.tStart = t  # local t and not account for scr refresh
                trial_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_text, 'tStartRefresh')  # time at next scr refresh
                # update status
                trial_text.status = STARTED
                trial_text.setAutoDraw(True)
            
            # if trial_text is active this frame...
            if trial_text.status == STARTED:
                # update params
                pass
            
            # if trial_text is stopping this frame...
            if trial_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > trial_text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    trial_text.tStop = t  # not accounting for scr refresh
                    trial_text.tStopRefresh = tThisFlipGlobal  # on global time
                    trial_text.frameNStop = frameN  # exact frame index
                    # update status
                    trial_text.status = FINISHED
                    trial_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial_Word_Display,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_Word_Display.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_Word_Display.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_Word_Display" ---
        for thisComponent in trial_Word_Display.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_Word_Display
        trial_Word_Display.tStop = globalClock.getTime(format='float')
        trial_Word_Display.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial_Word_Display.maxDurationReached:
            routineTimer.addTime(-trial_Word_Display.maxDuration)
        elif trial_Word_Display.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        # mark thisTrials_word as finished
        if hasattr(thisTrials_word, 'status'):
            thisTrials_word.status = FINISHED
        # if awaiting a pause, pause now
        if trials_words.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials_words.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_words'
    trials_words.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "clock_start" ---
    # create an object to store info about Routine clock_start
    clock_start = data.Routine(
        name='clock_start',
        components=[],
    )
    clock_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_clock_trial
    global_clock.reset()  # Start the global timer  
    
    # store start times for clock_start
    clock_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clock_start.tStart = globalClock.getTime(format='float')
    clock_start.status = STARTED
    clock_start.maxDuration = None
    # keep track of which components have finished
    clock_startComponents = clock_start.components
    for thisComponent in clock_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "clock_start" ---
    clock_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clock_start,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            clock_start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in clock_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clock_start" ---
    for thisComponent in clock_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clock_start
    clock_start.tStop = globalClock.getTime(format='float')
    clock_start.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "clock_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    recall_trial = data.TrialHandler2(
        name='recall_trial',
        nReps=6.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(recall_trial)  # add the loop to the experiment
    thisRecall_trial = recall_trial.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecall_trial.rgb)
    if thisRecall_trial != None:
        for paramName in thisRecall_trial:
            globals()[paramName] = thisRecall_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRecall_trial in recall_trial:
        recall_trial.status = STARTED
        if hasattr(thisRecall_trial, 'status'):
            thisRecall_trial.status = STARTED
        currentLoop = recall_trial
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRecall_trial.rgb)
        if thisRecall_trial != None:
            for paramName in thisRecall_trial:
                globals()[paramName] = thisRecall_trial[paramName]
        
        # --- Prepare to start Routine "recall_Trial" ---
        # create an object to store info about Routine recall_Trial
        recall_Trial = data.Routine(
            name='recall_Trial',
            components=[textbox_recall_Trial, key_resp_recall_Trial, mouse_trial],
        )
        recall_Trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textbox_recall_Trial.reset()
        textbox_recall_Trial.setText('')
        textbox_recall_Trial.setPlaceholder('')
        # create starting attributes for key_resp_recall_Trial
        key_resp_recall_Trial.keys = []
        key_resp_recall_Trial.rt = []
        _key_resp_recall_Trial_allKeys = []
        # setup some python lists for storing info about the mouse_trial
        gotValidClick = False  # until a click is received
        # store start times for recall_Trial
        recall_Trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_Trial.tStart = globalClock.getTime(format='float')
        recall_Trial.status = STARTED
        recall_Trial.maxDuration = None
        win.color = [-1.0000, 0.0039, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        recall_TrialComponents = recall_Trial.components
        for thisComponent in recall_Trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_Trial" ---
        recall_Trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRecall_trial, 'status') and thisRecall_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_recall_Trial
            # Add time elapsed in THIS iteration to total_time  
            total_time = global_clock.getTime()  
            # Terminate the loop if total_time exceeds 40 seconds  
            if total_time >= time_limit:  
                recall_trial.finished = True  # Force the loop to exit  
                continueRoutine = False  # Exit the current routine  
            
            if mouse_trial.getPressed()[0]:  # 0 = left, 1 = right, 2 = middle
                continueRoutine = False  # Exit the routine
                recall_trial.finished = True  # Force the loop to exit  
                print("Mouse skip clicked! Routine terminated.")
            
            
            
            # *textbox_recall_Trial* updates
            
            # if textbox_recall_Trial is starting this frame...
            if textbox_recall_Trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textbox_recall_Trial.frameNStart = frameN  # exact frame index
                textbox_recall_Trial.tStart = t  # local t and not account for scr refresh
                textbox_recall_Trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textbox_recall_Trial, 'tStartRefresh')  # time at next scr refresh
                # update status
                textbox_recall_Trial.status = STARTED
                textbox_recall_Trial.setAutoDraw(True)
            
            # if textbox_recall_Trial is active this frame...
            if textbox_recall_Trial.status == STARTED:
                # update params
                pass
            
            # *key_resp_recall_Trial* updates
            waitOnFlip = False
            
            # if key_resp_recall_Trial is starting this frame...
            if key_resp_recall_Trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_recall_Trial.frameNStart = frameN  # exact frame index
                key_resp_recall_Trial.tStart = t  # local t and not account for scr refresh
                key_resp_recall_Trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_recall_Trial, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_recall_Trial.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_recall_Trial.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_recall_Trial.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_recall_Trial.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_recall_Trial.getKeys(keyList=['return','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_recall_Trial_allKeys.extend(theseKeys)
                if len(_key_resp_recall_Trial_allKeys):
                    key_resp_recall_Trial.keys = _key_resp_recall_Trial_allKeys[-1].name  # just the last key pressed
                    key_resp_recall_Trial.rt = _key_resp_recall_Trial_allKeys[-1].rt
                    key_resp_recall_Trial.duration = _key_resp_recall_Trial_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # *mouse_trial* updates
            
            # if mouse_trial is starting this frame...
            if mouse_trial.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_trial.frameNStart = frameN  # exact frame index
                mouse_trial.tStart = t  # local t and not account for scr refresh
                mouse_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_trial, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_trial.status = STARTED
                mouse_trial.mouseClock.reset()
                prevButtonState = mouse_trial.getPressed()  # if button is down already this ISN'T a new click
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=recall_Trial,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_Trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_Trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_Trial" ---
        for thisComponent in recall_Trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_Trial
        recall_Trial.tStop = globalClock.getTime(format='float')
        recall_Trial.tStopRefresh = tThisFlipGlobal
        setupWindow(expInfo=expInfo, win=win)
        recall_trial.addData('textbox_recall_Trial.text',textbox_recall_Trial.text)
        # check responses
        if key_resp_recall_Trial.keys in ['', [], None]:  # No response was made
            key_resp_recall_Trial.keys = None
        recall_trial.addData('key_resp_recall_Trial.keys',key_resp_recall_Trial.keys)
        if key_resp_recall_Trial.keys != None:  # we had a response
            recall_trial.addData('key_resp_recall_Trial.rt', key_resp_recall_Trial.rt)
            recall_trial.addData('key_resp_recall_Trial.duration', key_resp_recall_Trial.duration)
        # store data for recall_trial (TrialHandler)
        # the Routine "recall_Trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisRecall_trial as finished
        if hasattr(thisRecall_trial, 'status'):
            thisRecall_trial.status = FINISHED
        # if awaiting a pause, pause now
        if recall_trial.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            recall_trial.status = STARTED
        thisExp.nextEntry()
        
    # completed 6.0 repeats of 'recall_trial'
    recall_trial.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "proceed_study" ---
    # create an object to store info about Routine proceed_study
    proceed_study = data.Routine(
        name='proceed_study',
        components=[bridge_Text, text_3],
    )
    proceed_study.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for proceed_study
    proceed_study.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    proceed_study.tStart = globalClock.getTime(format='float')
    proceed_study.status = STARTED
    thisExp.addData('proceed_study.started', proceed_study.tStart)
    proceed_study.maxDuration = None
    # keep track of which components have finished
    proceed_studyComponents = proceed_study.components
    for thisComponent in proceed_study.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "proceed_study" ---
    proceed_study.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *bridge_Text* updates
        
        # if bridge_Text is starting this frame...
        if bridge_Text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            bridge_Text.frameNStart = frameN  # exact frame index
            bridge_Text.tStart = t  # local t and not account for scr refresh
            bridge_Text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(bridge_Text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'bridge_Text.started')
            # update status
            bridge_Text.status = STARTED
            bridge_Text.setAutoDraw(True)
        
        # if bridge_Text is active this frame...
        if bridge_Text.status == STARTED:
            # update params
            pass
        
        # if bridge_Text is stopping this frame...
        if bridge_Text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > bridge_Text.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                bridge_Text.tStop = t  # not accounting for scr refresh
                bridge_Text.tStopRefresh = tThisFlipGlobal  # on global time
                bridge_Text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bridge_Text.stopped')
                # update status
                bridge_Text.status = FINISHED
                bridge_Text.setAutoDraw(False)
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 5.5-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # if text_3 is stopping this frame...
        if text_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_3.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                text_3.tStop = t  # not accounting for scr refresh
                text_3.tStopRefresh = tThisFlipGlobal  # on global time
                text_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.stopped')
                # update status
                text_3.status = FINISHED
                text_3.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=proceed_study,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            proceed_study.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in proceed_study.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "proceed_study" ---
    for thisComponent in proceed_study.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for proceed_study
    proceed_study.tStop = globalClock.getTime(format='float')
    proceed_study.tStopRefresh = tThisFlipGlobal
    thisExp.addData('proceed_study.stopped', proceed_study.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if proceed_study.maxDurationReached:
        routineTimer.addTime(-proceed_study.maxDuration)
    elif proceed_study.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.500000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    list1_words = data.TrialHandler2(
        name='list1_words',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'word_list_main.xlsx', 
        selection='6:24'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(list1_words)  # add the loop to the experiment
    thisList1_word = list1_words.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisList1_word.rgb)
    if thisList1_word != None:
        for paramName in thisList1_word:
            globals()[paramName] = thisList1_word[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisList1_word in list1_words:
        list1_words.status = STARTED
        if hasattr(thisList1_word, 'status'):
            thisList1_word.status = STARTED
        currentLoop = list1_words
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisList1_word.rgb)
        if thisList1_word != None:
            for paramName in thisList1_word:
                globals()[paramName] = thisList1_word[paramName]
        
        # --- Prepare to start Routine "study_list1" ---
        # create an object to store info about Routine study_list1
        study_list1 = data.Routine(
            name='study_list1',
            components=[text],
        )
        study_list1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text.setText(Words)
        # store start times for study_list1
        study_list1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        study_list1.tStart = globalClock.getTime(format='float')
        study_list1.status = STARTED
        study_list1.maxDuration = None
        # keep track of which components have finished
        study_list1Components = study_list1.components
        for thisComponent in study_list1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "study_list1" ---
        study_list1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # if trial has changed, end Routine now
            if hasattr(thisList1_word, 'status') and thisList1_word.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=study_list1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                study_list1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in study_list1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "study_list1" ---
        for thisComponent in study_list1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for study_list1
        study_list1.tStop = globalClock.getTime(format='float')
        study_list1.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if study_list1.maxDurationReached:
            routineTimer.addTime(-study_list1.maxDuration)
        elif study_list1.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        # mark thisList1_word as finished
        if hasattr(thisList1_word, 'status'):
            thisList1_word.status = FINISHED
        # if awaiting a pause, pause now
        if list1_words.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            list1_words.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'list1_words'
    list1_words.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "clock_start_list_1" ---
    # create an object to store info about Routine clock_start_list_1
    clock_start_list_1 = data.Routine(
        name='clock_start_list_1',
        components=[],
    )
    clock_start_list_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_clock_list_1
    global_clock.reset()  # Start the global timer  
    
    # store start times for clock_start_list_1
    clock_start_list_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clock_start_list_1.tStart = globalClock.getTime(format='float')
    clock_start_list_1.status = STARTED
    clock_start_list_1.maxDuration = None
    # keep track of which components have finished
    clock_start_list_1Components = clock_start_list_1.components
    for thisComponent in clock_start_list_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "clock_start_list_1" ---
    clock_start_list_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clock_start_list_1,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            clock_start_list_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in clock_start_list_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clock_start_list_1" ---
    for thisComponent in clock_start_list_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clock_start_list_1
    clock_start_list_1.tStop = globalClock.getTime(format='float')
    clock_start_list_1.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "clock_start_list_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    recall_list_1 = data.TrialHandler2(
        name='recall_list_1',
        nReps=18.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(recall_list_1)  # add the loop to the experiment
    thisRecall_list_1 = recall_list_1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecall_list_1.rgb)
    if thisRecall_list_1 != None:
        for paramName in thisRecall_list_1:
            globals()[paramName] = thisRecall_list_1[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRecall_list_1 in recall_list_1:
        recall_list_1.status = STARTED
        if hasattr(thisRecall_list_1, 'status'):
            thisRecall_list_1.status = STARTED
        currentLoop = recall_list_1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRecall_list_1.rgb)
        if thisRecall_list_1 != None:
            for paramName in thisRecall_list_1:
                globals()[paramName] = thisRecall_list_1[paramName]
        
        # --- Prepare to start Routine "recall_list1" ---
        # create an object to store info about Routine recall_list1
        recall_list1 = data.Routine(
            name='recall_list1',
            components=[list1_recall, key_resp_list_1_recall, mouse_1],
        )
        recall_list1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        list1_recall.reset()
        list1_recall.setText('')
        list1_recall.setPlaceholder('')
        # create starting attributes for key_resp_list_1_recall
        key_resp_list_1_recall.keys = []
        key_resp_list_1_recall.rt = []
        _key_resp_list_1_recall_allKeys = []
        # setup some python lists for storing info about the mouse_1
        gotValidClick = False  # until a click is received
        # store start times for recall_list1
        recall_list1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_list1.tStart = globalClock.getTime(format='float')
        recall_list1.status = STARTED
        recall_list1.maxDuration = None
        win.color = [-1.0000, 0.0039, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        recall_list1Components = recall_list1.components
        for thisComponent in recall_list1.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_list1" ---
        recall_list1.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRecall_list_1, 'status') and thisRecall_list_1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *list1_recall* updates
            
            # if list1_recall is starting this frame...
            if list1_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                list1_recall.frameNStart = frameN  # exact frame index
                list1_recall.tStart = t  # local t and not account for scr refresh
                list1_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(list1_recall, 'tStartRefresh')  # time at next scr refresh
                # update status
                list1_recall.status = STARTED
                list1_recall.setAutoDraw(True)
            
            # if list1_recall is active this frame...
            if list1_recall.status == STARTED:
                # update params
                pass
            
            # *key_resp_list_1_recall* updates
            waitOnFlip = False
            
            # if key_resp_list_1_recall is starting this frame...
            if key_resp_list_1_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_list_1_recall.frameNStart = frameN  # exact frame index
                key_resp_list_1_recall.tStart = t  # local t and not account for scr refresh
                key_resp_list_1_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_list_1_recall, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_list_1_recall.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_list_1_recall.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_list_1_recall.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_list_1_recall.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_list_1_recall.getKeys(keyList=['return','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_list_1_recall_allKeys.extend(theseKeys)
                if len(_key_resp_list_1_recall_allKeys):
                    key_resp_list_1_recall.keys = _key_resp_list_1_recall_allKeys[-1].name  # just the last key pressed
                    key_resp_list_1_recall.rt = _key_resp_list_1_recall_allKeys[-1].rt
                    key_resp_list_1_recall.duration = _key_resp_list_1_recall_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from code_list_1_recall
            # Add time elapsed in THIS iteration to total_time  
            total_time = global_clock.getTime()  
            # Terminate the loop if total_time exceeds 40 seconds  
            if total_time >= time_limit:  
                recall_list_1.finished = True  # Force the loop to exit  
                continueRoutine = False  # Exit the current routine  
            
            if mouse_1.getPressed()[0]:  # 0 = left, 1 = right, 2 = middle
                continueRoutine = False  # Exit the routine
                recall_list_1.finished = True  # Force the loop to exit  
                print("Mouse skip clicked! Routine terminated.")
            
            
            # *mouse_1* updates
            
            # if mouse_1 is starting this frame...
            if mouse_1.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_1.frameNStart = frameN  # exact frame index
                mouse_1.tStart = t  # local t and not account for scr refresh
                mouse_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_1, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_1.status = STARTED
                mouse_1.mouseClock.reset()
                prevButtonState = mouse_1.getPressed()  # if button is down already this ISN'T a new click
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=recall_list1,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_list1.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_list1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_list1" ---
        for thisComponent in recall_list1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_list1
        recall_list1.tStop = globalClock.getTime(format='float')
        recall_list1.tStopRefresh = tThisFlipGlobal
        setupWindow(expInfo=expInfo, win=win)
        recall_list_1.addData('list1_recall.text',list1_recall.text)
        # check responses
        if key_resp_list_1_recall.keys in ['', [], None]:  # No response was made
            key_resp_list_1_recall.keys = None
        recall_list_1.addData('key_resp_list_1_recall.keys',key_resp_list_1_recall.keys)
        if key_resp_list_1_recall.keys != None:  # we had a response
            recall_list_1.addData('key_resp_list_1_recall.rt', key_resp_list_1_recall.rt)
            recall_list_1.addData('key_resp_list_1_recall.duration', key_resp_list_1_recall.duration)
        # store data for recall_list_1 (TrialHandler)
        # the Routine "recall_list1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisRecall_list_1 as finished
        if hasattr(thisRecall_list_1, 'status'):
            thisRecall_list_1.status = FINISHED
        # if awaiting a pause, pause now
        if recall_list_1.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            recall_list_1.status = STARTED
        thisExp.nextEntry()
        
    # completed 18.0 repeats of 'recall_list_1'
    recall_list_1.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "proceed_bridge" ---
    # create an object to store info about Routine proceed_bridge
    proceed_bridge = data.Routine(
        name='proceed_bridge',
        components=[text_4],
    )
    proceed_bridge.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for proceed_bridge
    proceed_bridge.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    proceed_bridge.tStart = globalClock.getTime(format='float')
    proceed_bridge.status = STARTED
    proceed_bridge.maxDuration = None
    # keep track of which components have finished
    proceed_bridgeComponents = proceed_bridge.components
    for thisComponent in proceed_bridge.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "proceed_bridge" ---
    proceed_bridge.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # if text_4 is stopping this frame...
        if text_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_4.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_4.tStop = t  # not accounting for scr refresh
                text_4.tStopRefresh = tThisFlipGlobal  # on global time
                text_4.frameNStop = frameN  # exact frame index
                # update status
                text_4.status = FINISHED
                text_4.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=proceed_bridge,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            proceed_bridge.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in proceed_bridge.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "proceed_bridge" ---
    for thisComponent in proceed_bridge.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for proceed_bridge
    proceed_bridge.tStop = globalClock.getTime(format='float')
    proceed_bridge.tStopRefresh = tThisFlipGlobal
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if proceed_bridge.maxDurationReached:
        routineTimer.addTime(-proceed_bridge.maxDuration)
    elif proceed_bridge.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    list2_words = data.TrialHandler2(
        name='list2_words',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'word_list_main.xlsx', 
        selection='24:42'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(list2_words)  # add the loop to the experiment
    thisList2_word = list2_words.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisList2_word.rgb)
    if thisList2_word != None:
        for paramName in thisList2_word:
            globals()[paramName] = thisList2_word[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisList2_word in list2_words:
        list2_words.status = STARTED
        if hasattr(thisList2_word, 'status'):
            thisList2_word.status = STARTED
        currentLoop = list2_words
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisList2_word.rgb)
        if thisList2_word != None:
            for paramName in thisList2_word:
                globals()[paramName] = thisList2_word[paramName]
        
        # --- Prepare to start Routine "study_list2" ---
        # create an object to store info about Routine study_list2
        study_list2 = data.Routine(
            name='study_list2',
            components=[text_2],
        )
        study_list2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_2.setText(Words)
        # store start times for study_list2
        study_list2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        study_list2.tStart = globalClock.getTime(format='float')
        study_list2.status = STARTED
        thisExp.addData('study_list2.started', study_list2.tStart)
        study_list2.maxDuration = None
        # keep track of which components have finished
        study_list2Components = study_list2.components
        for thisComponent in study_list2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "study_list2" ---
        study_list2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # if trial has changed, end Routine now
            if hasattr(thisList2_word, 'status') and thisList2_word.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # if text_2 is stopping this frame...
            if text_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_2.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_2.tStop = t  # not accounting for scr refresh
                    text_2.tStopRefresh = tThisFlipGlobal  # on global time
                    text_2.frameNStop = frameN  # exact frame index
                    # update status
                    text_2.status = FINISHED
                    text_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=study_list2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                study_list2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in study_list2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "study_list2" ---
        for thisComponent in study_list2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for study_list2
        study_list2.tStop = globalClock.getTime(format='float')
        study_list2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('study_list2.stopped', study_list2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if study_list2.maxDurationReached:
            routineTimer.addTime(-study_list2.maxDuration)
        elif study_list2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        # mark thisList2_word as finished
        if hasattr(thisList2_word, 'status'):
            thisList2_word.status = FINISHED
        # if awaiting a pause, pause now
        if list2_words.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            list2_words.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'list2_words'
    list2_words.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "suffix_play" ---
    # create an object to store info about Routine suffix_play
    suffix_play = data.Routine(
        name='suffix_play',
        components=[sound_1],
    )
    suffix_play.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    sound_1.setSound('laminer48.wav', hamming=True)
    sound_1.setVolume(1.0, log=False)
    sound_1.seek(0)
    # store start times for suffix_play
    suffix_play.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    suffix_play.tStart = globalClock.getTime(format='float')
    suffix_play.status = STARTED
    thisExp.addData('suffix_play.started', suffix_play.tStart)
    suffix_play.maxDuration = None
    # keep track of which components have finished
    suffix_playComponents = suffix_play.components
    for thisComponent in suffix_play.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "suffix_play" ---
    suffix_play.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *sound_1* updates
        
        # if sound_1 is starting this frame...
        if sound_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sound_1.frameNStart = frameN  # exact frame index
            sound_1.tStart = t  # local t and not account for scr refresh
            sound_1.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            sound_1.status = STARTED
            sound_1.play(when=win)  # sync with win flip
        
        # if sound_1 is stopping this frame...
        if sound_1.status == STARTED:
            if bool(False) or sound_1.isFinished:
                # keep track of stop time/frame for later
                sound_1.tStop = t  # not accounting for scr refresh
                sound_1.tStopRefresh = tThisFlipGlobal  # on global time
                sound_1.frameNStop = frameN  # exact frame index
                # update status
                sound_1.status = FINISHED
                sound_1.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=suffix_play,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            suffix_play.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in suffix_play.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "suffix_play" ---
    for thisComponent in suffix_play.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for suffix_play
    suffix_play.tStop = globalClock.getTime(format='float')
    suffix_play.tStopRefresh = tThisFlipGlobal
    thisExp.addData('suffix_play.stopped', suffix_play.tStop)
    sound_1.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "suffix_play" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "clock_start_list_2" ---
    # create an object to store info about Routine clock_start_list_2
    clock_start_list_2 = data.Routine(
        name='clock_start_list_2',
        components=[],
    )
    clock_start_list_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_clock_list
    global_clock.reset()  # Start the global timer  
    
    # store start times for clock_start_list_2
    clock_start_list_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clock_start_list_2.tStart = globalClock.getTime(format='float')
    clock_start_list_2.status = STARTED
    clock_start_list_2.maxDuration = None
    # keep track of which components have finished
    clock_start_list_2Components = clock_start_list_2.components
    for thisComponent in clock_start_list_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "clock_start_list_2" ---
    clock_start_list_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clock_start_list_2,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            clock_start_list_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in clock_start_list_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clock_start_list_2" ---
    for thisComponent in clock_start_list_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clock_start_list_2
    clock_start_list_2.tStop = globalClock.getTime(format='float')
    clock_start_list_2.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "clock_start_list_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    recall_list_2 = data.TrialHandler2(
        name='recall_list_2',
        nReps=18.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(recall_list_2)  # add the loop to the experiment
    thisRecall_list_2 = recall_list_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecall_list_2.rgb)
    if thisRecall_list_2 != None:
        for paramName in thisRecall_list_2:
            globals()[paramName] = thisRecall_list_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRecall_list_2 in recall_list_2:
        recall_list_2.status = STARTED
        if hasattr(thisRecall_list_2, 'status'):
            thisRecall_list_2.status = STARTED
        currentLoop = recall_list_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRecall_list_2.rgb)
        if thisRecall_list_2 != None:
            for paramName in thisRecall_list_2:
                globals()[paramName] = thisRecall_list_2[paramName]
        
        # --- Prepare to start Routine "recall_list2" ---
        # create an object to store info about Routine recall_list2
        recall_list2 = data.Routine(
            name='recall_list2',
            components=[list2_recall, key_resp_list_1_recall_2, mouse],
        )
        recall_list2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        list2_recall.reset()
        list2_recall.setText('')
        list2_recall.setPlaceholder('')
        # create starting attributes for key_resp_list_1_recall_2
        key_resp_list_1_recall_2.keys = []
        key_resp_list_1_recall_2.rt = []
        _key_resp_list_1_recall_2_allKeys = []
        # setup some python lists for storing info about the mouse
        gotValidClick = False  # until a click is received
        # store start times for recall_list2
        recall_list2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_list2.tStart = globalClock.getTime(format='float')
        recall_list2.status = STARTED
        thisExp.addData('recall_list2.started', recall_list2.tStart)
        recall_list2.maxDuration = None
        win.color = [-1.0000, 0.0039, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        recall_list2Components = recall_list2.components
        for thisComponent in recall_list2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_list2" ---
        recall_list2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRecall_list_2, 'status') and thisRecall_list_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *list2_recall* updates
            
            # if list2_recall is starting this frame...
            if list2_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                list2_recall.frameNStart = frameN  # exact frame index
                list2_recall.tStart = t  # local t and not account for scr refresh
                list2_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(list2_recall, 'tStartRefresh')  # time at next scr refresh
                # update status
                list2_recall.status = STARTED
                list2_recall.setAutoDraw(True)
            
            # if list2_recall is active this frame...
            if list2_recall.status == STARTED:
                # update params
                pass
            
            # *key_resp_list_1_recall_2* updates
            waitOnFlip = False
            
            # if key_resp_list_1_recall_2 is starting this frame...
            if key_resp_list_1_recall_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_list_1_recall_2.frameNStart = frameN  # exact frame index
                key_resp_list_1_recall_2.tStart = t  # local t and not account for scr refresh
                key_resp_list_1_recall_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_list_1_recall_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_list_1_recall_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_list_1_recall_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_list_1_recall_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_list_1_recall_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_list_1_recall_2.getKeys(keyList=['return','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_list_1_recall_2_allKeys.extend(theseKeys)
                if len(_key_resp_list_1_recall_2_allKeys):
                    key_resp_list_1_recall_2.keys = _key_resp_list_1_recall_2_allKeys[-1].name  # just the last key pressed
                    key_resp_list_1_recall_2.rt = _key_resp_list_1_recall_2_allKeys[-1].rt
                    key_resp_list_1_recall_2.duration = _key_resp_list_1_recall_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from code_list_1_recall_2
            # Add time elapsed in THIS iteration to total_time  
            total_time = global_clock.getTime()  
            # Terminate the loop if total_time exceeds 40 seconds  
            if total_time >= time_limit:  
                recall_list_2.finished = True  # Force the loop to exit  
                continueRoutine = False  # Exit the current routine  
            
            if mouse.getPressed()[0]:  # 0 = left, 1 = right, 2 = middle
                continueRoutine = False  # Exit the routine
                recall_list_2.finished = True  # Force the loop to exit  
                print("Mouse skip clicked! Routine terminated.")
            
            
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse.status = STARTED
                mouse.mouseClock.reset()
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=recall_list2,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_list2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_list2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_list2" ---
        for thisComponent in recall_list2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_list2
        recall_list2.tStop = globalClock.getTime(format='float')
        recall_list2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recall_list2.stopped', recall_list2.tStop)
        setupWindow(expInfo=expInfo, win=win)
        recall_list_2.addData('list2_recall.text',list2_recall.text)
        # check responses
        if key_resp_list_1_recall_2.keys in ['', [], None]:  # No response was made
            key_resp_list_1_recall_2.keys = None
        recall_list_2.addData('key_resp_list_1_recall_2.keys',key_resp_list_1_recall_2.keys)
        if key_resp_list_1_recall_2.keys != None:  # we had a response
            recall_list_2.addData('key_resp_list_1_recall_2.rt', key_resp_list_1_recall_2.rt)
            recall_list_2.addData('key_resp_list_1_recall_2.duration', key_resp_list_1_recall_2.duration)
        # store data for recall_list_2 (TrialHandler)
        # the Routine "recall_list2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisRecall_list_2 as finished
        if hasattr(thisRecall_list_2, 'status'):
            thisRecall_list_2.status = FINISHED
        # if awaiting a pause, pause now
        if recall_list_2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            recall_list_2.status = STARTED
        thisExp.nextEntry()
        
    # completed 18.0 repeats of 'recall_list_2'
    recall_list_2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "proceed_bridge_2" ---
    # create an object to store info about Routine proceed_bridge_2
    proceed_bridge_2 = data.Routine(
        name='proceed_bridge_2',
        components=[text_6],
    )
    proceed_bridge_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for proceed_bridge_2
    proceed_bridge_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    proceed_bridge_2.tStart = globalClock.getTime(format='float')
    proceed_bridge_2.status = STARTED
    proceed_bridge_2.maxDuration = None
    # keep track of which components have finished
    proceed_bridge_2Components = proceed_bridge_2.components
    for thisComponent in proceed_bridge_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "proceed_bridge_2" ---
    proceed_bridge_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_6* updates
        
        # if text_6 is starting this frame...
        if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_6.frameNStart = frameN  # exact frame index
            text_6.tStart = t  # local t and not account for scr refresh
            text_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_6.status = STARTED
            text_6.setAutoDraw(True)
        
        # if text_6 is active this frame...
        if text_6.status == STARTED:
            # update params
            pass
        
        # if text_6 is stopping this frame...
        if text_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_6.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_6.tStop = t  # not accounting for scr refresh
                text_6.tStopRefresh = tThisFlipGlobal  # on global time
                text_6.frameNStop = frameN  # exact frame index
                # update status
                text_6.status = FINISHED
                text_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=proceed_bridge_2,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            proceed_bridge_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in proceed_bridge_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "proceed_bridge_2" ---
    for thisComponent in proceed_bridge_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for proceed_bridge_2
    proceed_bridge_2.tStop = globalClock.getTime(format='float')
    proceed_bridge_2.tStopRefresh = tThisFlipGlobal
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if proceed_bridge_2.maxDurationReached:
        routineTimer.addTime(-proceed_bridge_2.maxDuration)
    elif proceed_bridge_2.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    list3_words = data.TrialHandler2(
        name='list3_words',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'word_list_main.xlsx', 
        selection='42:60'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(list3_words)  # add the loop to the experiment
    thisList3_word = list3_words.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisList3_word.rgb)
    if thisList3_word != None:
        for paramName in thisList3_word:
            globals()[paramName] = thisList3_word[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisList3_word in list3_words:
        list3_words.status = STARTED
        if hasattr(thisList3_word, 'status'):
            thisList3_word.status = STARTED
        currentLoop = list3_words
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisList3_word.rgb)
        if thisList3_word != None:
            for paramName in thisList3_word:
                globals()[paramName] = thisList3_word[paramName]
        
        # --- Prepare to start Routine "study_list3" ---
        # create an object to store info about Routine study_list3
        study_list3 = data.Routine(
            name='study_list3',
            components=[text_5],
        )
        study_list3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_5.setText(Words)
        # store start times for study_list3
        study_list3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        study_list3.tStart = globalClock.getTime(format='float')
        study_list3.status = STARTED
        study_list3.maxDuration = None
        # keep track of which components have finished
        study_list3Components = study_list3.components
        for thisComponent in study_list3.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "study_list3" ---
        study_list3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # if trial has changed, end Routine now
            if hasattr(thisList3_word, 'status') and thisList3_word.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_5* updates
            
            # if text_5 is starting this frame...
            if text_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_5.frameNStart = frameN  # exact frame index
                text_5.tStart = t  # local t and not account for scr refresh
                text_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_5.status = STARTED
                text_5.setAutoDraw(True)
            
            # if text_5 is active this frame...
            if text_5.status == STARTED:
                # update params
                pass
            
            # if text_5 is stopping this frame...
            if text_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_5.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_5.tStop = t  # not accounting for scr refresh
                    text_5.tStopRefresh = tThisFlipGlobal  # on global time
                    text_5.frameNStop = frameN  # exact frame index
                    # update status
                    text_5.status = FINISHED
                    text_5.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=study_list3,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                study_list3.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in study_list3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "study_list3" ---
        for thisComponent in study_list3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for study_list3
        study_list3.tStop = globalClock.getTime(format='float')
        study_list3.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if study_list3.maxDurationReached:
            routineTimer.addTime(-study_list3.maxDuration)
        elif study_list3.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        # mark thisList3_word as finished
        if hasattr(thisList3_word, 'status'):
            thisList3_word.status = FINISHED
        # if awaiting a pause, pause now
        if list3_words.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            list3_words.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'list3_words'
    list3_words.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "clock_start_list_3" ---
    # create an object to store info about Routine clock_start_list_3
    clock_start_list_3 = data.Routine(
        name='clock_start_list_3',
        components=[],
    )
    clock_start_list_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_clock_list_2
    global_clock.reset()  # Start the global timer  
    
    # store start times for clock_start_list_3
    clock_start_list_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clock_start_list_3.tStart = globalClock.getTime(format='float')
    clock_start_list_3.status = STARTED
    clock_start_list_3.maxDuration = None
    # keep track of which components have finished
    clock_start_list_3Components = clock_start_list_3.components
    for thisComponent in clock_start_list_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "clock_start_list_3" ---
    clock_start_list_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clock_start_list_3,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            clock_start_list_3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in clock_start_list_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clock_start_list_3" ---
    for thisComponent in clock_start_list_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clock_start_list_3
    clock_start_list_3.tStop = globalClock.getTime(format='float')
    clock_start_list_3.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "clock_start_list_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    recall_list_3 = data.TrialHandler2(
        name='recall_list_3',
        nReps=18.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(recall_list_3)  # add the loop to the experiment
    thisRecall_list_3 = recall_list_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecall_list_3.rgb)
    if thisRecall_list_3 != None:
        for paramName in thisRecall_list_3:
            globals()[paramName] = thisRecall_list_3[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRecall_list_3 in recall_list_3:
        recall_list_3.status = STARTED
        if hasattr(thisRecall_list_3, 'status'):
            thisRecall_list_3.status = STARTED
        currentLoop = recall_list_3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRecall_list_3.rgb)
        if thisRecall_list_3 != None:
            for paramName in thisRecall_list_3:
                globals()[paramName] = thisRecall_list_3[paramName]
        
        # --- Prepare to start Routine "recall_list3" ---
        # create an object to store info about Routine recall_list3
        recall_list3 = data.Routine(
            name='recall_list3',
            components=[list3_recall, key_resp_list_1_recall_3, mouse_3],
        )
        recall_list3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        list3_recall.reset()
        list3_recall.setText('')
        list3_recall.setPlaceholder('')
        # create starting attributes for key_resp_list_1_recall_3
        key_resp_list_1_recall_3.keys = []
        key_resp_list_1_recall_3.rt = []
        _key_resp_list_1_recall_3_allKeys = []
        # setup some python lists for storing info about the mouse_3
        gotValidClick = False  # until a click is received
        # store start times for recall_list3
        recall_list3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_list3.tStart = globalClock.getTime(format='float')
        recall_list3.status = STARTED
        thisExp.addData('recall_list3.started', recall_list3.tStart)
        recall_list3.maxDuration = None
        win.color = [-1.0000, 0.0039, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        recall_list3Components = recall_list3.components
        for thisComponent in recall_list3.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_list3" ---
        recall_list3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRecall_list_3, 'status') and thisRecall_list_3.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *list3_recall* updates
            
            # if list3_recall is starting this frame...
            if list3_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                list3_recall.frameNStart = frameN  # exact frame index
                list3_recall.tStart = t  # local t and not account for scr refresh
                list3_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(list3_recall, 'tStartRefresh')  # time at next scr refresh
                # update status
                list3_recall.status = STARTED
                list3_recall.setAutoDraw(True)
            
            # if list3_recall is active this frame...
            if list3_recall.status == STARTED:
                # update params
                pass
            
            # *key_resp_list_1_recall_3* updates
            waitOnFlip = False
            
            # if key_resp_list_1_recall_3 is starting this frame...
            if key_resp_list_1_recall_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_list_1_recall_3.frameNStart = frameN  # exact frame index
                key_resp_list_1_recall_3.tStart = t  # local t and not account for scr refresh
                key_resp_list_1_recall_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_list_1_recall_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_list_1_recall_3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_list_1_recall_3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_list_1_recall_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_list_1_recall_3.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_list_1_recall_3.getKeys(keyList=['return','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_list_1_recall_3_allKeys.extend(theseKeys)
                if len(_key_resp_list_1_recall_3_allKeys):
                    key_resp_list_1_recall_3.keys = _key_resp_list_1_recall_3_allKeys[-1].name  # just the last key pressed
                    key_resp_list_1_recall_3.rt = _key_resp_list_1_recall_3_allKeys[-1].rt
                    key_resp_list_1_recall_3.duration = _key_resp_list_1_recall_3_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from code_list_1_recall_3
            # Add time elapsed in THIS iteration to total_time  
            total_time = global_clock.getTime()  
            # Terminate the loop if total_time exceeds 40 seconds  
            if total_time >= time_limit:  
                recall_list_3.finished = True  # Force the loop to exit  
                continueRoutine = False  # Exit the current routine  
            
            if mouse.getPressed()[0]:  # 0 = left, 1 = right, 2 = middle
                continueRoutine = False  # Exit the routine
                recall_list_3.finished = True  # Force the loop to exit  
                print("Mouse skip clicked! Routine terminated.")
            
            
            # *mouse_3* updates
            
            # if mouse_3 is starting this frame...
            if mouse_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_3.frameNStart = frameN  # exact frame index
                mouse_3.tStart = t  # local t and not account for scr refresh
                mouse_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_3.status = STARTED
                mouse_3.mouseClock.reset()
                prevButtonState = mouse_3.getPressed()  # if button is down already this ISN'T a new click
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=recall_list3,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_list3.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_list3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_list3" ---
        for thisComponent in recall_list3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_list3
        recall_list3.tStop = globalClock.getTime(format='float')
        recall_list3.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recall_list3.stopped', recall_list3.tStop)
        setupWindow(expInfo=expInfo, win=win)
        recall_list_3.addData('list3_recall.text',list3_recall.text)
        # check responses
        if key_resp_list_1_recall_3.keys in ['', [], None]:  # No response was made
            key_resp_list_1_recall_3.keys = None
        recall_list_3.addData('key_resp_list_1_recall_3.keys',key_resp_list_1_recall_3.keys)
        if key_resp_list_1_recall_3.keys != None:  # we had a response
            recall_list_3.addData('key_resp_list_1_recall_3.rt', key_resp_list_1_recall_3.rt)
            recall_list_3.addData('key_resp_list_1_recall_3.duration', key_resp_list_1_recall_3.duration)
        # store data for recall_list_3 (TrialHandler)
        # the Routine "recall_list3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisRecall_list_3 as finished
        if hasattr(thisRecall_list_3, 'status'):
            thisRecall_list_3.status = FINISHED
        # if awaiting a pause, pause now
        if recall_list_3.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            recall_list_3.status = STARTED
        thisExp.nextEntry()
        
    # completed 18.0 repeats of 'recall_list_3'
    recall_list_3.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "proceed_birdge_3" ---
    # create an object to store info about Routine proceed_birdge_3
    proceed_birdge_3 = data.Routine(
        name='proceed_birdge_3',
        components=[text_7],
    )
    proceed_birdge_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for proceed_birdge_3
    proceed_birdge_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    proceed_birdge_3.tStart = globalClock.getTime(format='float')
    proceed_birdge_3.status = STARTED
    proceed_birdge_3.maxDuration = None
    # keep track of which components have finished
    proceed_birdge_3Components = proceed_birdge_3.components
    for thisComponent in proceed_birdge_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "proceed_birdge_3" ---
    proceed_birdge_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_7* updates
        
        # if text_7 is starting this frame...
        if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_7.frameNStart = frameN  # exact frame index
            text_7.tStart = t  # local t and not account for scr refresh
            text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_7.status = STARTED
            text_7.setAutoDraw(True)
        
        # if text_7 is active this frame...
        if text_7.status == STARTED:
            # update params
            pass
        
        # if text_7 is stopping this frame...
        if text_7.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_7.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_7.tStop = t  # not accounting for scr refresh
                text_7.tStopRefresh = tThisFlipGlobal  # on global time
                text_7.frameNStop = frameN  # exact frame index
                # update status
                text_7.status = FINISHED
                text_7.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=proceed_birdge_3,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            proceed_birdge_3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in proceed_birdge_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "proceed_birdge_3" ---
    for thisComponent in proceed_birdge_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for proceed_birdge_3
    proceed_birdge_3.tStop = globalClock.getTime(format='float')
    proceed_birdge_3.tStopRefresh = tThisFlipGlobal
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if proceed_birdge_3.maxDurationReached:
        routineTimer.addTime(-proceed_birdge_3.maxDuration)
    elif proceed_birdge_3.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    list4_words = data.TrialHandler2(
        name='list4_words',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'word_list_main.xlsx', 
        selection='60:78'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(list4_words)  # add the loop to the experiment
    thisList4_word = list4_words.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisList4_word.rgb)
    if thisList4_word != None:
        for paramName in thisList4_word:
            globals()[paramName] = thisList4_word[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisList4_word in list4_words:
        list4_words.status = STARTED
        if hasattr(thisList4_word, 'status'):
            thisList4_word.status = STARTED
        currentLoop = list4_words
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisList4_word.rgb)
        if thisList4_word != None:
            for paramName in thisList4_word:
                globals()[paramName] = thisList4_word[paramName]
        
        # --- Prepare to start Routine "study_list4" ---
        # create an object to store info about Routine study_list4
        study_list4 = data.Routine(
            name='study_list4',
            components=[text_8],
        )
        study_list4.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_8.setText(Words)
        # store start times for study_list4
        study_list4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        study_list4.tStart = globalClock.getTime(format='float')
        study_list4.status = STARTED
        thisExp.addData('study_list4.started', study_list4.tStart)
        study_list4.maxDuration = None
        # keep track of which components have finished
        study_list4Components = study_list4.components
        for thisComponent in study_list4.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "study_list4" ---
        study_list4.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # if trial has changed, end Routine now
            if hasattr(thisList4_word, 'status') and thisList4_word.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_8* updates
            
            # if text_8 is starting this frame...
            if text_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_8.frameNStart = frameN  # exact frame index
                text_8.tStart = t  # local t and not account for scr refresh
                text_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_8.started')
                # update status
                text_8.status = STARTED
                text_8.setAutoDraw(True)
            
            # if text_8 is active this frame...
            if text_8.status == STARTED:
                # update params
                pass
            
            # if text_8 is stopping this frame...
            if text_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_8.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_8.tStop = t  # not accounting for scr refresh
                    text_8.tStopRefresh = tThisFlipGlobal  # on global time
                    text_8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_8.stopped')
                    # update status
                    text_8.status = FINISHED
                    text_8.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=study_list4,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                study_list4.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in study_list4.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "study_list4" ---
        for thisComponent in study_list4.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for study_list4
        study_list4.tStop = globalClock.getTime(format='float')
        study_list4.tStopRefresh = tThisFlipGlobal
        thisExp.addData('study_list4.stopped', study_list4.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if study_list4.maxDurationReached:
            routineTimer.addTime(-study_list4.maxDuration)
        elif study_list4.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        # mark thisList4_word as finished
        if hasattr(thisList4_word, 'status'):
            thisList4_word.status = FINISHED
        # if awaiting a pause, pause now
        if list4_words.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            list4_words.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'list4_words'
    list4_words.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "clock_start_list_4" ---
    # create an object to store info about Routine clock_start_list_4
    clock_start_list_4 = data.Routine(
        name='clock_start_list_4',
        components=[],
    )
    clock_start_list_4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_clock_list_3
    global_clock.reset()  # Start the global timer  
    
    # store start times for clock_start_list_4
    clock_start_list_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    clock_start_list_4.tStart = globalClock.getTime(format='float')
    clock_start_list_4.status = STARTED
    clock_start_list_4.maxDuration = None
    # keep track of which components have finished
    clock_start_list_4Components = clock_start_list_4.components
    for thisComponent in clock_start_list_4.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "clock_start_list_4" ---
    clock_start_list_4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=clock_start_list_4,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            clock_start_list_4.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in clock_start_list_4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "clock_start_list_4" ---
    for thisComponent in clock_start_list_4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for clock_start_list_4
    clock_start_list_4.tStop = globalClock.getTime(format='float')
    clock_start_list_4.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "clock_start_list_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "suffix_play_2" ---
    # create an object to store info about Routine suffix_play_2
    suffix_play_2 = data.Routine(
        name='suffix_play_2',
        components=[sound_2],
    )
    suffix_play_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    sound_2.setSound('aktuator48.wav', hamming=True)
    sound_2.setVolume(1.0, log=False)
    sound_2.seek(0)
    # store start times for suffix_play_2
    suffix_play_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    suffix_play_2.tStart = globalClock.getTime(format='float')
    suffix_play_2.status = STARTED
    suffix_play_2.maxDuration = None
    # keep track of which components have finished
    suffix_play_2Components = suffix_play_2.components
    for thisComponent in suffix_play_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "suffix_play_2" ---
    suffix_play_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *sound_2* updates
        
        # if sound_2 is starting this frame...
        if sound_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sound_2.frameNStart = frameN  # exact frame index
            sound_2.tStart = t  # local t and not account for scr refresh
            sound_2.tStartRefresh = tThisFlipGlobal  # on global time
            # update status
            sound_2.status = STARTED
            sound_2.play(when=win)  # sync with win flip
        
        # if sound_2 is stopping this frame...
        if sound_2.status == STARTED:
            if bool(False) or sound_2.isFinished:
                # keep track of stop time/frame for later
                sound_2.tStop = t  # not accounting for scr refresh
                sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                sound_2.frameNStop = frameN  # exact frame index
                # update status
                sound_2.status = FINISHED
                sound_2.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=suffix_play_2,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            suffix_play_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in suffix_play_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "suffix_play_2" ---
    for thisComponent in suffix_play_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for suffix_play_2
    suffix_play_2.tStop = globalClock.getTime(format='float')
    suffix_play_2.tStopRefresh = tThisFlipGlobal
    sound_2.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "suffix_play_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    recall_list_4 = data.TrialHandler2(
        name='recall_list_4',
        nReps=18.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'word_list_main.xlsx', 
        selection='61:78'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(recall_list_4)  # add the loop to the experiment
    thisRecall_list_4 = recall_list_4.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRecall_list_4.rgb)
    if thisRecall_list_4 != None:
        for paramName in thisRecall_list_4:
            globals()[paramName] = thisRecall_list_4[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRecall_list_4 in recall_list_4:
        recall_list_4.status = STARTED
        if hasattr(thisRecall_list_4, 'status'):
            thisRecall_list_4.status = STARTED
        currentLoop = recall_list_4
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRecall_list_4.rgb)
        if thisRecall_list_4 != None:
            for paramName in thisRecall_list_4:
                globals()[paramName] = thisRecall_list_4[paramName]
        
        # --- Prepare to start Routine "recall_list4" ---
        # create an object to store info about Routine recall_list4
        recall_list4 = data.Routine(
            name='recall_list4',
            components=[list4_recall, key_resp_list_1_recall_4, mouse_4],
        )
        recall_list4.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        list4_recall.reset()
        list4_recall.setText('')
        list4_recall.setPlaceholder('')
        # create starting attributes for key_resp_list_1_recall_4
        key_resp_list_1_recall_4.keys = []
        key_resp_list_1_recall_4.rt = []
        _key_resp_list_1_recall_4_allKeys = []
        # setup some python lists for storing info about the mouse_4
        gotValidClick = False  # until a click is received
        # store start times for recall_list4
        recall_list4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_list4.tStart = globalClock.getTime(format='float')
        recall_list4.status = STARTED
        recall_list4.maxDuration = None
        win.color = [-1.0000, 0.0039, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        recall_list4Components = recall_list4.components
        for thisComponent in recall_list4.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_list4" ---
        recall_list4.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisRecall_list_4, 'status') and thisRecall_list_4.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *list4_recall* updates
            
            # if list4_recall is starting this frame...
            if list4_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                list4_recall.frameNStart = frameN  # exact frame index
                list4_recall.tStart = t  # local t and not account for scr refresh
                list4_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(list4_recall, 'tStartRefresh')  # time at next scr refresh
                # update status
                list4_recall.status = STARTED
                list4_recall.setAutoDraw(True)
            
            # if list4_recall is active this frame...
            if list4_recall.status == STARTED:
                # update params
                pass
            
            # *key_resp_list_1_recall_4* updates
            waitOnFlip = False
            
            # if key_resp_list_1_recall_4 is starting this frame...
            if key_resp_list_1_recall_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_list_1_recall_4.frameNStart = frameN  # exact frame index
                key_resp_list_1_recall_4.tStart = t  # local t and not account for scr refresh
                key_resp_list_1_recall_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_list_1_recall_4, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_list_1_recall_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_list_1_recall_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_list_1_recall_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_list_1_recall_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_list_1_recall_4.getKeys(keyList=['return','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_list_1_recall_4_allKeys.extend(theseKeys)
                if len(_key_resp_list_1_recall_4_allKeys):
                    key_resp_list_1_recall_4.keys = _key_resp_list_1_recall_4_allKeys[-1].name  # just the last key pressed
                    key_resp_list_1_recall_4.rt = _key_resp_list_1_recall_4_allKeys[-1].rt
                    key_resp_list_1_recall_4.duration = _key_resp_list_1_recall_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from code_list_1_recall_4
            # Add time elapsed in THIS iteration to total_time  
            total_time = global_clock.getTime()  
            # Terminate the loop if total_time exceeds 40 seconds  
            if total_time >= time_limit:  
                recall_list_4.finished = True  # Force the loop to exit  
                continueRoutine = False  # Exit the current routine  
            
            if mouse_4.getPressed()[0]:  # 0 = left, 1 = right, 2 = middle
                continueRoutine = False  # Exit the routine
                recall_list_4.finished = True  # Force the loop to exit  
                print("Mouse skip clicked! Routine terminated.")
            
            
            # *mouse_4* updates
            
            # if mouse_4 is starting this frame...
            if mouse_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_4.frameNStart = frameN  # exact frame index
                mouse_4.tStart = t  # local t and not account for scr refresh
                mouse_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_4, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_4.status = STARTED
                mouse_4.mouseClock.reset()
                prevButtonState = mouse_4.getPressed()  # if button is down already this ISN'T a new click
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=recall_list4,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_list4.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_list4.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_list4" ---
        for thisComponent in recall_list4.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_list4
        recall_list4.tStop = globalClock.getTime(format='float')
        recall_list4.tStopRefresh = tThisFlipGlobal
        setupWindow(expInfo=expInfo, win=win)
        recall_list_4.addData('list4_recall.text',list4_recall.text)
        # check responses
        if key_resp_list_1_recall_4.keys in ['', [], None]:  # No response was made
            key_resp_list_1_recall_4.keys = None
        recall_list_4.addData('key_resp_list_1_recall_4.keys',key_resp_list_1_recall_4.keys)
        if key_resp_list_1_recall_4.keys != None:  # we had a response
            recall_list_4.addData('key_resp_list_1_recall_4.rt', key_resp_list_1_recall_4.rt)
            recall_list_4.addData('key_resp_list_1_recall_4.duration', key_resp_list_1_recall_4.duration)
        # store data for recall_list_4 (TrialHandler)
        # the Routine "recall_list4" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisRecall_list_4 as finished
        if hasattr(thisRecall_list_4, 'status'):
            thisRecall_list_4.status = FINISHED
        # if awaiting a pause, pause now
        if recall_list_4.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            recall_list_4.status = STARTED
        thisExp.nextEntry()
        
    # completed 18.0 repeats of 'recall_list_4'
    recall_list_4.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[text_end],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end* updates
        
        # if text_end is starting this frame...
        if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end.frameNStart = frameN  # exact frame index
            text_end.tStart = t  # local t and not account for scr refresh
            text_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end.started')
            # update status
            text_end.status = STARTED
            text_end.setAutoDraw(True)
        
        # if text_end is active this frame...
        if text_end.status == STARTED:
            # update params
            pass
        
        # if text_end is stopping this frame...
        if text_end.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_end.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_end.tStop = t  # not accounting for scr refresh
                text_end.tStopRefresh = tThisFlipGlobal  # on global time
                text_end.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_end.stopped')
                # update status
                text_end.status = FINISHED
                text_end.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=end,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end.maxDurationReached:
        routineTimer.addTime(-end.maxDuration)
    elif end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from code_script_run
    import subprocess 
    
    experimentPath = os.getcwd() 
    scriptName = "hello.py"
    
    scriptPath = os.path.join(experimentPath, scriptName)
    
    subproces.run(["python",scriptPath])
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)

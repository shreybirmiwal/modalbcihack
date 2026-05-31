# Idea

The project is a model that inputs live brain eeg data and then outputs browser use actions (move mouse, click, etc). We will take the browser use actiosn and demo this on soemthing like a videogame.


Since the hackathon is karpathy autoresearch hackathon, the model will be trained using karpathy autoresearch -- we purposefully choose a task that can be both easy (easy game for demo) and hard (like a full browseruse) so we can say that eventually auto research will crack the hard, dream goal. Also, Auto research is for problems that have a massive search space for parameters and training models, which is perfect.

### model inputs
the bci hardware code is 
sdk for the BCI: https://github.com/fs-re-ak/AlchemiacPythonSDK#
since the cssv data doesnt have timestamps, we will stream using lsl library and save the data


### training
we need to 


### model output
we want some discrete outputs for simple mvp
 - shooting/clicking --> lets do this with muscle wet electrode on arm, when u squeeze arm
 - jumping/space bar --> lets do other muscle wet electrode on other arm
 jaw clench = move forward / forward arrow key
 focus with brain with brain it will turn right -- itll keep turing for as long as u focus. to move left you have to focus for long enough for it to get past 180*


this means we need 1 wet electrode on left arm, 1 wet electrode on right arm, 1 copper near jaw, focus will be more 

### infernce time 
stream data from electrodes using:https://github.com/fs-re-ak/AlchemiacPythonSDK#
forwards data to lsl

### live game inference

The live demo uses the Modal/autoresearch model exported at
`auto research/runs/final_model.json`.

Start the headset stream in one terminal:

```bash
cd bci-sdk
uv run AlchemiacStreamLSL.py
```

Then start the live inference game controller in another terminal:

```bash
cd bci-sdk
uv run AlchemiacGameController.py
```

The game controller keeps the wave visualizer on the left and shows game tabs
on the right:

- Flappy Bird: `eye_blink` flaps. Space is a keyboard fallback.
- Pong: `right_squeeze` moves the paddle up, `left_squeeze` moves it down.
- Crossy Road: `left_squeeze` moves forward.
- Geometry Dash: `right_squeeze` jumps.

You can test without the headset by replaying a saved CSV through the same
model and game path:

```bash
cd bci-sdk
uv run AlchemiacGameController.py --replay-csv data/blink_prod.csv
```

The older recorder is still available when you only want CSV capture and wave
marking:

```bash
uv run AlchemiacController.py
```


# hackathon details
this is the modal autoresearch hackathon
they want: ​We’re looking for teams ready to tackle research problems in data- or compute-intensive domains.
​What to build:

​Agent Architectures & Control Loops – innovate on the core systems design of autonomous research agents. Improve performance and scalability by advancing planning, memory, coordination and control.

​Retrieval & Knowledge Synthesis – build systems that excel at finding, validating, and integrating information at scale. Develop pipelines for large-scale data ingestion, advanced retrieval, citation and structured knowledge extraction.

​Applied Autonomous Research – implement an end-to-end autoresearch agent tailored to a high-impact domain. We’re especially interested in applications in law, computational biology, coding and beyond.

What We’re Looking For	
Technical Depth	Strong systems engineering, architecture, scalability
Originality	Novel ideas or frontier approaches
Demo Clarity	Easy to understand, clear explanation, working demo
Standout Execution in one of three factors 	• Agent Architectures & Control Loops – innovate on the core systems design of autonomous research agents. Improve performance and scalability by advancing planning, memory, coordination and control.
• Retrieval & Knowledge Synthesis – build systems that excel at finding, validating, and integrating information at scale. Develop pipelines for large-scale data ingestion, advanced retrieval, citation and structured knowledge extraction.
• Applied Autonomous Research – implement an end-to-end autoresearch agent tailored to a high-impact domain. We’re especially interested in applications in law, computational biology, coding and beyond.

#### Relevant docs:

- https://modal.com/blog/autoscaling-autoresearch
- https://modal.com/blog/building-with-modal-and-the-openai-agent-sdk


### 3. demo



### the BCI headset we have
-[] Alchemiac board  (MCU + BLE + ads1299 + 9 axis IMU), that's the same board found in the Hermes
-[] u.FL mezzanine breakout board, used to connect with shielded cables and provides access to 2 extra ADC + several GPIOs (from the MCU) - you need to modify firmware to use them.
-[] Header pin mezzanine breakout board, used to connect with "OpenBCI style electrodes" and provides access to 2 extra ADC + several GPIOs (from the MCU) - you need to modify firmware to use them.
-[] One batterie charging module (battery not included for shipping purpose, but any lipo will do, buy it on amazon, I'll share a model)
-[] A few cables to connect it all.
-[] 1 basic 3d printed box

- One set of 8 shielded cables (need to agree on length)
- One set of OpenBCI style electrodes (gold cup)
- 8 PCB electrodes with u.fl connector (screwable, with gold contact dry electrodes) + screws
- 8 PCB flex BCI anything electrodes with u.fl connector (cuttable, glueable pcb flex to be attached to whatever, gold contact dry electrodes)

**Not included: **MCU programmer. The device is pre-programmed, but if you want to play with the firmware (open source | github), you'll need a programmer.

* With this kit, you'll be able to build one or two shielded biosensing system out of anything (headphones, glasses, glove, armband) and depending of you hacking skills.
* You'll have an 8 wet electrodes setup (same as OpenBCI) for standard EEG readings
* open source python SDK with viewer (open source | github)
* closed source android app for recording




## Auto research setup

- `auto research/`: Python autoresearch engine, frozen eval harness, Modal CPU
  runner, and program instructions.
- `web/`: React/Vite dashboard for visualizing candidate runs and improvement
  ladders.

## Quick Start

```bash
cd "auto research"
python3 loop.py --subject S03 --stage 2 --rounds 3 --batch-size 10 --workers 4
```

```bash
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

# VECA Documentation

## Contents

* [Python-side API Doc](#python-side-api-doc)
* [Unity-side API Doc](#unity-side-api-doc)
* [Directory Structure](#directory-structure)

### Python side API Doc

VECA.environment.GeneralEnvironment

Environment class which connects VECA-Unity application with VECA-python.

| Attribute | Type | Description |
| --- | --- | --- |
| port | int | Port of the python server. |
| num\_envs | int | Number of environments in VECA-unity application. |

| Function | Type | Description |
| --- | --- | --- |
| \_\_init\_\_(self, NUM\_AGENTS, port) | void | Initialize this class object. Assign NUM\_AGENTS to self.num\_envs, port to self.port. |
| start\_connection(self) | void | Open up the socket connection. If the socket is open, it will print &#39;0.0.0.0&#39;. If the connection is established, it will print &quot;CONNECTED&quot;, self.num\_envs and print &quot;GO&quot;. |
| send\_action(self, action) | void | Send action to the application. _action_ must be an array which has size of _self.num\_envs_ x _ACTION\_LENGTH_. (ACTION\_LENGTH is dimension of the action defined in environment.) |
| reset(self) | void | Restart the application. In specific, it calls AgentReset() defined in _VECA.GeneralHeadQuarter_. |
| reset\_connection(self) | void | Automatically restart VECA-unity application and restore the connection. It is useful when VECA-unity application suffers from problems such as memory leak. |
| close(self) | void | Closes the connection. |

### Unity side API Doc

VECA.GeneralAgent

Controls single agent in this unity application.

| Attribute | Type | Description |
| --- | --- | --- |
| agent | public GameObject | Agent controlled by the script. |
| environment | public GameObject | Environment of the agent. |
| GeneralAgentCharObs | public Dictionary\&lt;string, List\&lt;char\&gt;\&gt; | Character type observation buffer. |
| GeneralAgentIntObs | public Dictionary\&lt;string, List\&lt;int\&gt;\&gt; | Integer type observation buffer. |
| GeneralAgentFloatObs | public Dictionary\&lt;string, List\&lt;float\&gt;\&gt; | Float type observation buffer. |
| GeneralAgentUInt8Obs | public Dictionary\&lt;string, List\&lt;byte\&gt;\&gt; | Uint8 type observation buffer. |
| GeneralAgentInt16Obs | public Dictionary\&lt;string, List\&lt;short\&gt;\&gt; | Int16 type observation buffer. |
| done | protected bool | Whether the simulation is over(e.g. end of the game). |
| reward | protected float | Current reward of the agent. |
| recorder | protected RecorderMyMP4 | Whether the simulation is over(e.g. end of the game). |

| Function | Type | Description |
| --- | --- | --- |
| GeneralAwake() | public void | Initialize this class object. |
| GetDone() | public byte[] | Convert the &#39;done&#39; variable to byte and return it. |
| ResetObservation() | public void | Clear the observation buffers mentioned above. |
| StartRecording(int num) | public void | DEPRECATED |
| EndRecording() | public void | DEPRECATED |
| AgentAction(float[] _action_) | public abstract void | Define the action of the agent corresponding to the _action_ vector. |
| AgentReset() | public abstract void | Reset the agent. |
| CollectObservations() | public abstract void | Collect all observations observed by the _agent_ controlled by this script. |

VECA.GeneralHeadQuarter

Controls all environment/agents in this unity application.

| Attribute | Type | Description |
| --- | --- | --- |
| agent | public GameObject | Sample agent controlled by the script. |
| environment | public GameObject | Sample environment of the agent. |
| agents | protected GeneralAgent[] | Agents controlled by the script.(Usually made by duplication of agent) |
| environments | protected GameObject[] | Environments controlled by the script.(Usually made by duplication of environment) |
| client | public ClientEnv | ClientEnv object which communicates with the python side. |
| action | public float[][] | Action of agents. |
| GeneralAgentCharObs | public Dictionary\&lt;string, List\&lt;char\&gt;\&gt; | Character type observation buffer. |
| GeneralAgentIntObs | public Dictionary\&lt;string, List\&lt;int\&gt;\&gt; | Integer type observation buffer. |
| GeneralAgentFloatObs | public Dictionary\&lt;string, List\&lt;float\&gt;\&gt; | Float type observation buffer. |
| GeneralAgentUInt8Obs | public Dictionary\&lt;string, List\&lt;byte\&gt;\&gt; | Uint8 type observation buffer. |
| GeneralAgentInt16Obs | public Dictionary\&lt;string, List\&lt;short\&gt;\&gt; | Int16 type observation buffer. |
| isRecording | protected boolean | DEPRECATED |
| NUM\_AGENTS | protected int | Number of agents = Number of environments |
| ACTION\_LENGTH | protected int | Dimension of action space = Length of _action_ vector in AgentAction() of GeneralAgent class. |
| simulationSpeed | protected float | Time distance between adjacent frames. Same to reciprocal of FPS(frame per second). |
| debug | protected boolean | True if debug mode. In debug mode, the environment doesn&#39;t try to connect with the python server. |
| fixedframe | protected boolean | True if the FPS is fixed. |
| minT | protected float | Time distance between adjacent physics update. |
| IP | protected String | IP of the python server. |
| PORT | protected int | Port of the python server. |
| isLocal | protected boolean | True if local mode. In local mode, the environment doesn&#39;t try to connect with the python server. |
| timeOut | protected int | Maximum waiting time in connection with the python server. |
| safeminT | private float | Minimum time distance between physics &amp; animation update for stable physics.(default 0.001) |

| Function | Type | Description |
| --- | --- | --- |
| getAgentsInEnv(Transform cur) | private static List\&lt;GeneralAgent\&gt; | Get all the agents in environment _cur_. |
| GeneralStart() | public void | Initialize this headquarter object. It allows to use various arguments while calling the unity app, which would be mentioned later. |
| GeneralUpdate() | public void | Updates a single frame of the environments and agents controlled by headquarter object(Similar to Update() in Unity). Further details would be mentioned later. |
| RunForwardPass() | public void | Update the physics and animations. |
| getAnimatorsInEnv(Transform cur) | public static List\&lt;GeneralAgent\&gt; | Get all the animators in environment _cur_. |
| GeneralStart() | public void | Initialize of this headquarter object. It allows to use various arguments while calling the unity app, which would be mentioned later. |
| SimulateAnimation(float _dt_) | public void | Update the animation for time distance _dt_. |
| CollectAgentsObservations() | public void | Collect observations for all agents in _agents,_ and store those in its observation buffer. |
| AgentsReset() | public void | Reset all agents in _agents._ |
| AgentsAction() | public void |   |

VECA.ClientEnv

A network-based client class which establishes and manages the connection between VECA-python side.

| Attribute | Type | Description |
| --- | --- | --- |
| socketConnection | public TcpClient | Socket connection between python server and this application. |
| stream | public NetworkStream | NetworkStream object used in connection. |
| IP | public string | IP of the python server. |
| PORT | public int | Port of the python server. |
| timeOut | public int | Maximum waiting time while connection in milliseconds. |

| Function | Type | Description |
| --- | --- | --- |
| ConnectToTcpServer() | public bool | Setup socket connection with the server. return true if connection is established. |
| ListenData(int _dataLength_) | public byte[] | Get _dataLength_ byte of data from python server. |
| SendData(byte[] _Message_) | public void | Send _message_ data to python server. |
| GetAction(int NUM\_AGENTS, int ACTION\_LENGTH) | public float[][] | Get action vectors for each agents. |
| SendObservation(Dictionary\&lt;string, List\&lt;T\&gt;[]\&gt; obs) | public void | Send observation _obs_ to python server. T is one of char, int, short, byte, float. |

VECA.Humanoid.VECAHumanoidObs

Abstract class which defines the format of observation for humanoid agents.

| Attribute | Type | Description |
| --- | --- | --- |

| Function | Type | Description |
| --- | --- | --- |
| GetImage() | public abstract float[] | Get image observation. MUST IMPLEMENT. |
| GetAudio() | public abstract float[] | Get audio observation. MUST IMPLEMENT. |
| GetTactile() | public abstract float[] | Get tactile observation. MUST IMPLEMENT. |
| GetProprioception() | public abstract float[] | Get proprioception observation. MUST IMPLEMENT. |

VECA.Humanoid.VECAHumanoidAgent : VECA.GeneralAgent

Controls single humanoid agent in this unity application. Supports useful features for humanoid agents.

| Attribute | Type | Description |
| --- | --- | --- |
| humanoidObs | protected VECAHumanoidObs | VECAHumanoidObs object which collects observations of the agent. |

| Function | Type | Description |
| --- | --- | --- |
| GetImage | public byte[] | Get image observation. |
| GetAudio | public short[] | Get audio observation. |
| GetTactile | public float[] | Get tactile observation. |
| VECAHumanoidAwake | public float[][] | Initialize this object. |
| getHumanoidObs | public VECAHumanoidObs | Return _humanoidObs_ of this object. |

VECA.Humanoid.VECAHumanoidHeadQuarter : VECA.GeneralHeadQuarter

Controls all environments which consist of humanoid agents.

| Attribute | Type | Description |
| --- | --- | --- |
| spatializer | public GameObject | Spatializer object which contains _ControlAudioListener_ script. |
| realistic | protected bool | True if audio data uses realistic filter effects. |

| Function | Type | Description |
| --- | --- | --- |
| VECAHumanoidUpdate() | protected void | Updates a single frame of the environments and agents controlled by headquarter object(Similar to Update() in Unity). Takes a same role with GeneralUpdate() of VECA.GeneralHeadQuarter. |
| VECAHumanoidStart() | public float[][] | Initialize this object. |

VECA.Humanoid.Example.VECAHumanoidExampleInteract

Example interactions of humanoid agents. This class supports movements and action such as walk, kick and grab, it also supports several utility functions such as lookTowards(). Please note that physical interactions(such as pushing with body) is possible without using any additional functions, but those interactions supposes that those interaction doesn&#39;t interrupt the agent&#39;s movement.(For example, tripping over object is impossible).

| Attribute | Type | Description |
| --- | --- | --- |
| agent | public GameObject | Agent controlled by the script. |
| leftHand | public GameObject | Left hand of the agent. |
| rightHand | public GameObject | Right hand of the agent. |
| leftFoot | public GameObject | Left foot of the agent. |
| rightFoot | public GameObject | Right foot of the agent. |
| head | public GameObject | Head of the agent. |
| eyeL | public GameObject | Left eye of the agent. |
| eyeR | public GameObject | Right eye of the agent. |
| cameraL | public GameObject | Camera assigned to left eye of the agent. |
| cameraR | public GameObject | Camera assigned to right eye of the agent. |
| anim | protected Animator | Animator of the agent. |
| grabbedObject | protected GameObject | Object which the agent is grabbing. Null if the agent is not grabbing anything. |
| grabLength | protected float | Distance between hand and the _grabbedObject._ |
| fixedLook | protected bool | True if direction of the agent is fixed. It makes VECA-Unity do not update the direction of camera while moving. |
| kicking | protected bool | True if agent is middle of kicking animation. It prevents the agent from kicking another object before current kicking action ends. |
| walking | protected bool | True if agent is walking. |
| walkSpeed | protected float | Walking speed of the agent. |
| turnSpeed | protected float | Turning speed(left, right) of the agent. |
| focalLength | protected float | Focal distance of _cameraL_ and _cameraR_. |

| Function | Type | Description |
| --- | --- | --- |
| walk(float walkFloat, float turnFloat) | public void | Make the agent walk in desired direction. In specific, it walks forward with the speed of _walkSpeed_ \* _walkFloat_, and turns direction with the angular speed of _walkSpeed_ \* _walkFloat._ |
| kick(GameObject obj, Vector3 force) | public void | Kick the object with the force of _force._ This function doesn&#39;t check whether the object is close enough to kick. |
| kick(GameObject obj) | public void | Kick the object with random force. |
| grab(GameObject obj, float grabDistance) | public void | Grab the object with the distance _grabDistance_ from the body. Grabbed object becomes kinematic while grabbed, and restores its physical properties when released. |
| release() | public void | Release the grabbed object. If the agent didn&#39;t grab anything, this function does nothing. |
| rotateUpDownHead(float deg) | public void | Rotate the head in the Up-Down plane. Head goes down when deg\&gt;0. Please note that this function doesn&#39;t have any constraints: Excessive rotation might distort the mesh of the agent. |
| rotateLeftRightHead(float deg) | public void | Rotate the head in the Left-right plane. Head goes right when deg\&gt;0. Please note that this function doesn&#39;t have any constraints: Excessive rotation might distort the mesh of the agent. |
| lookTowardPoint(Vector3 point) | public void | Make the agent look at 3-dimensional point. Note that this function doesn&#39;t rotate the head : it only rotates the camera(eye). Also, the agent will look at the point until releaseTowardPoint() is called, although agent is moving. |
| releaseTowardPoint(Vector3 point) | public void | Release the perspective lock and make the agent see forward. |
| adjustFocalLength(float newFocalLength) | public void | Adjust the focal distance of the cameras. (Due to limits of unity post-processing units, currently VECA-unity does not support different focal length between two eyes) |
| makeSound(GameObject obj) | public void | Make the object sound if the object has an audio source inside. The object will play the sound until the audio ends or _stopSound(obj)_ is called. |
| stopSound(GameObject obj) | public void | Make the object sound if the object has an audio source inside. |

VECA.Humanoid.Example.VECAHumanoidExampleObs : VECA.Humanoid.HumanoidObs

Example observations for humanoid agents. This class supports binocular vision, 3D-spatialized realistic sounds, float-valued &amp; normalized tactile data.

| Attribute | Type | Description |
| --- | --- | --- |
| env | public GameObject | Agent controlled by the script. |
| cameraL | public GameObject | Left hand of the agent. |
| cameraR | public GameObject | Right hand of the agent. |
| head | public GameObject | Left foot of the agent. |
| earL | public GameObject | Right foot of the agent. |
| earR | public GameObject | Head of the agent. |
| meshCollider | public GameObject | Left eye of the agent. |

| Function | Type | Description |
| --- | --- | --- |
| GetImage() | public abstract float[] | Get binocular image observation. |
| GetAudio() | public abstract float[] | Get 3D-spatialized audio observation. |
| GetTactile() | public abstract float[] | Get tactile observation. If the function is called, the function draws arrows on the editor which represent the forces. Please note that tactile calculation involves heavy computational procedures. |
| GetProprioception() | public abstract float[] | CURRENTLY NOT IMPLEMENTED |

 VECA.ObservationUtils(static class)

A static class which provides useful utility functions in calculating observations.

| Function | Type | Description |
| --- | --- | --- |
| getImage(Camera cam, int imgHeight, int imgWidth, bool grayscale) | public static float[] | Capture the image with the size of _imgHeight_ x _imgWidth_ from the camera _cam_. |
| getSource(Transform cur) | public static List\&lt;AudioSource\&gt; | Find all audio sources in the environment _cur._ |
| getAudio(Transform transform, List\&lt;AudioSource\&gt; sources) | public static float[] | Returns binaural audio data from the agent&#39;s _transform_ generated by audio sources in _sources,_ based on only distance. |
| getAudio(Transform transform, GameObject env) | public static float[] | Returns binaural audio data from the agent&#39;s _transform_ generated by audio sources in environment _env,_ based on only distance. |
| getSpatialAudio(Transform head, Vector3 earLposition, Vector3 earRposition, List\&lt;AudioSource\&gt; sources) | public static float[] | Returns binaural audio data from the agent&#39;s head orientation and position of both ears, generated by audio sources in _sources._ This audio data is spatialized using LISTEN dataset. |
| getSpatialAudio(Transform head, Vector3 earLposition, Vector3 earRposition, GameObject env) | public static float[] | Returns binaural audio data from the agent&#39;s head orientation and position of both ears, generated by audio sources in environment _env._ This audio data is spatialized using LISTEN dataset. |
| getSpatialAudio(Transform head, Vector3 earLposition, Vector3 earRposition, List\&lt;AudioSource\&gt; sources, Vector3 roomSize, float beta) | public static float[] | Returns binaural audio data from the agent&#39;s head orientation and position of both ears, generated by audio sources in _sources._ This audio data is spatialized using LISTEN dataset and reverb effect is applied using RIR generator. |
| getSpatialAudio(Transform head, Vector3 earLposition, Vector3 earRposition, GameObject env, Vector3 roomSize, float beta) | public static float[] | Returns binaural audio data from the agent&#39;s head orientation and position of both ears, generated by audio sources in environment _env._ This audio data is spatialized using LISTEN dataset and reverb effect is applied using RIR generator. |
| getTactile(VECATaxel[] taxels) | public static float[] | Returns tactile data using _taxels_. Forces are normalized with their maximum measurable force. |
| getTactile(MeshCollider mesh) | public static float[] | Returns tactile data using _mesh_ of the agent. Forces are normalized with their maximum measurable force. |

### Directory Structure 

```
VECA
├── python      // OpenAI Gym-like Environment APIs for Cognitive Agents
│   ├── environment.py
│   └── Example
│       └── KickTheBallEnv.py
└── unity       // VECA and Training Environment on Unity
    ├── ClientEnv.cs
    ├── Example
    │   ├── KickTheBallAgent.cs
    │   ├── KickTheBallHQ.cs
    │   ├── KickTheBallInteract.cs
    │   └── Shooter.cs
    ├── EyesController.cs
    ├── GeneralAgent.cs
    ├── GeneralAnimator.cs
    ├── GeneralHeadQuarter.cs
    ├── RecorderMyMp4.cs
    ├── VECAHumanoid
    │   ├── Audio
    │   │   ├── ControlAudioListener.cs
    │   │   ├── MySpatializer.cs
    │   │   └── ObservationUtils.cs
    │   ├── Tactile
    │   │   ├── VECATactile.cs
    │   │   └── VECATaxel.cs
    │   ├── VECAHumanoidAgent.cs
    │   ├── VECAHumanoidExample
    │   │   ├── 31.fbx
    │   │   ├── Baby.prefab
    │   │   ├── mainCameraFollow.cs
    │   │   ├── VECAHumanoidExampleCollision.cs
    │   │   ├── VECAHumanoidExampleInteract.cs
    │   │   └── VECAHumanoidExampleObs.cs
    │   ├── VECAHumanoidHeadQuarter.cs
    │   ├── VECAHumanoidObs.cs
    │   └── Vision
    │       ├── BlackNWhite.cs
    │       ├── BNW.shader
    │       ├── effect.cs
    │       ├── FocusPuller.cs
    │       └── TextureScale.cs
    └── VECATime.cs
```




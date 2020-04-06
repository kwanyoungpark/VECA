# VECA-unity Tutorial
## Headquarter

Did you have some fun? Now, let's see what's actually going on. Let's navigate to KickTheBallHQ.cs file, which is in the Assets/Examples/KickTheBall/Scripts folder. 

```cs
public class KickTheBallHQ : VECAHumanoidHeadQuarter
{
    void Start()
    {
        Screen.fullScreen = false;
        ACTION_LENGTH = 2;  
        VECAHumanoidStart();
        simulationSpeed = 1.0F / 15;
        VECATime.fixedDeltaTime = simulationSpeed;
        minT = 1.0F / 60;
        Application.targetFrameRate = 30;
    }

    void Update()
    {
        VECAHumanoidUpdate();
        //Thread.Sleep(25);
    }
}
```

Although the script looks simple, you may not underestimate its power. This code controls the entire application, which can even contain multiple parallel environments. In specific, VECAHumanoidStart() initialize and configure the settings of the environments, while VECAHumanoidUpdate() simulates the environment continuously.

Thanks to the dirty works handled by those functions, you can make VECA-unity environment much easier. Make an control script which inherits *GeneralAgent* class, implement 3 functions : AgentReset(), AgentAction(action), CollectObservations(). These functions define how to reset, action of the agent, observation of the agent.

Let's take a closer look about these function through examples. Navigate to KickTheBallAgent.cs file, which is in the Assets/Examples/KickTheBall/Scripts folder. It inherits VECAHumanoidAgent class, which also inherits *GeneralAgent* class.

## Reset

Let's focus on AgentReset() function.

```cs
public override void AgentReset()
{
    agent.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
    float deg = UnityEngine.Random.Range(0.0F, 360.0F);
    agent.transform.localRotation = Quaternion.AngleAxis(deg, new Vector3(0, 1, 0));
    agent.transform.position = environment.transform.position + new Vector3(UnityEngine.Random.Range(-1.0F, 1.0F), 0, UnityEngine.Random.Range(-1.0F, 1.0F)) * 5;
    controller.shooter.GetComponent<Shooter>().StartShooting();
}
```

AgentReset() defines how to reset the environment. In the example, it randomizes the position of the agent and reset the shooters.

## Action

Let's focus on AgentAction() function.

```cs	
public override void AgentAction(float[] action)
{
    if (done)
    {
        AgentReset();
        done = false;
    }
    for (int i = 0; i < action.Length; i++)
    {
        Debug.Log("Action " + Convert.ToString(i) + ": " + action[i].ToString());
    }
    // do something
    controller.work("walk", action[0], action[1]);
}
```

It simply converts the action vector to an action of the agent. In detail, action[0] sets the walking speed of the agent, and action[1] sets the direction of the agent. You can investigate the details and implementation of the *work*(String *key*, float *walkfloat*, float *turnFloat*) function in *KickTheBallInteract.cs*, which is located in the same folder.

## Observations

Let's focus on CollectObservations() function.

```cs
public override void CollectObservations()
{
    Vector3 myBabyStartPos = environment.transform.position;
    if (Math.Abs(agent.transform.position.z - myBabyStartPos.z) > 9.5 || Math.Abs(agent.transform.position.x - myBabyStartPos.x) > 9.5 ||
        Math.Abs(controller.ball.transform.position.z - myBabyStartPos.z) > 9.5 || Math.Abs(controller.ball.transform.position.x - myBabyStartPos.x) > 9.5)
    {
        done = true;
        AgentReset();
    }
    ResetObservation();
    AddObservation("img", GetImage());
    AddObservation("wav", GetAudio());
    AddObservation("done", GetDone());
    AddObservation("pos", relativePos);
    relativePos = controller.getRelativePos();
    reward = controller.getReward();
    AddObservation("reward", new float[] { reward });
    controller.reward = 0;
}
```

As you can see, *CollectObservations()* function calls *ResetObservation()* first, and call bunch of *AddObservation(String key, T[] observation)* function. There is an internal buffer to store observations in the GeneralAgent class. You can clear the buffer with *ResetObservation()* and add observation using *AddObservation(String key, T[] observation)*. (Please note that VECA-unity currently supports only 5 types(char[], int[], byte[], short[], float[]) of observation.)

 You can investigate the details and implementation of *GetImage(), GetAudio()* functions in *VECAHumanoidExampleObs.cs*, which is located in Assets/VECAScript/VECAHumanoid/VECAHumanoidExample folder. You might want to take a closer look of *ObservationUtils.cs", which is located in Assets/VECAScript/VECAHumanoid folder.

## More Environments

If you are not sufficient, you can explore more features of VECA-unity in another scene. Navigate to Assets/MainEnvironment folder and open *MainEnvironment* scene. You can do much more things such as grab, rotate head, even adjust focal distances!

In specific, you can grab/release the ball with pressing G, and adjust the grab distance with up/down arrow. If the agent is grabbing nothing, you can rotate agent's head by up/down/left/right arrow. Also you can make the ball play/stop its sound by pressing P. Please note that each features could be implemented with only 3~5 lines thanks to utility functions provided in *VECAHumanoidExampleInteract.cs*.



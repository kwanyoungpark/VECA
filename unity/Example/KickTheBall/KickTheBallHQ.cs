using UnityEngine;
using VECA.Humanoid;
using VECA;

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


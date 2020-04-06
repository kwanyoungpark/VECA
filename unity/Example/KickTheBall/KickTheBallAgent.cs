using UnityEngine;
using System;
using VECA.Humanoid;

public class KickTheBallAgent : VECAHumanoidAgent
{
    private KickTheBallInteract controller;
    Vector3 relativePos;
    /* 190726:1644 EJ : BirdEyeView Recorder Added */

    void Awake()
    {
        VECAHumanoidAwake(); 
        controller = agent.GetComponent<KickTheBallInteract>();
        relativePos = controller.getRelativePos();
    }

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

    public void Update()
    {

    }

    public override void AgentReset()
    {
        agent.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        float deg = UnityEngine.Random.Range(0.0F, 360.0F);
        agent.transform.localRotation = Quaternion.AngleAxis(deg, new Vector3(0, 1, 0));
        agent.transform.position = environment.transform.position + new Vector3(UnityEngine.Random.Range(-1.0F, 1.0F), 0, UnityEngine.Random.Range(-1.0F, 1.0F)) * 5;
        controller.shooter.GetComponent<Shooter>().StartShooting();
    }

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

}
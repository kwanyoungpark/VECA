using UnityEngine;
using VECA;
using VECA.Humanoid.Example;

public class KickTheBallInteract : VECAHumanoidExampleInteract
{
    public GameObject ball;
    public GameObject shooter;

    public bool isCollide;
    private Vector3 ballv, ballx, balla;

    float ballDistance;
    float limit = 2f;
    //bool onceFlag;
    bool triggered;
    public float reward;
    private int T;
    public float distance = 1.85f;

    string prevClipName = "Idle";
    float nextFrame;
    bool ballKicking;

    public bool eyesFollowMouse;

    // Start is called before the first frame update
    void Start()
    {
        base.Start();
        anim.SetFloat("walk", 0f);
        anim.SetFloat("turn", 0f);
        walkSpeed = 3f;
        turnSpeed = 80f;
        shooter.GetComponent<Shooter>().StartShooting();
    }

    public void work(string key)
    {
        if (key == "left")
        {
            Debug.Log("key 'left' is given.");
            kicking = false;
            agent.transform.Rotate(0, -120f * VECATime.fixedDeltaTime, 0);
        }
        else if (key == "right")
        {
            Debug.Log("key 'right' is given.");
            kicking = false;
            agent.transform.Rotate(0, 120f * VECATime.fixedDeltaTime, 0);
        }
        else if (key == "kick")
        {
            Debug.Log("key 'kick' is given.");
            if (!anim.GetCurrentAnimatorStateInfo(0).IsName("Kick Up"))
            {
                kicking = true;
            }
        }
    }

    public void work(string key, float walkFloat, float turnFloat) //function for agent
    {
        if (key == "walk")
        {
            Debug.Log("key 'walk' is given : " + walkFloat.ToString() + " " + turnFloat.ToString());
            kicking = false;
            if (!triggered)
            {
                walk(walkFloat, turnFloat);
                agent.transform.localPosition += walkFloat * agent.transform.forward * walkSpeed * VECATime.fixedDeltaTime;
                /*//fixedFramedAnimate("Blend Tree", VECATime.fixedDeltaTime);
                agent.transform.localPosition += walkFloat * agent.transform.forward * walkSpeed * VECATime.fixedDeltaTime;
                if (turnFloat < 0) {
                    agent.transform.RotateAround(leftFoot.transform.position, Vector3.up, turnFloat * turnSpeed * VECATime.fixedDeltaTime);
                }
                else
                {
                    agent.transform.RotateAround(rightFoot.transform.position, Vector3.up, turnFloat * turnSpeed * VECATime.fixedDeltaTime);
                }
                anim.SetFloat("walk", Mathf.Clamp(walkFloat, -1, 1));
                anim.SetFloat("turn", Mathf.Clamp(turnFloat, -1, 1));*/
            }
        }
    }

    public float getReward()
    {
        return reward;
    }

    // Update is called once per frame
    void Update()
    {
        base.Update();
        if (Input.GetKey(KeyCode.W) == true) walk(1f, 0f);
        else if (Input.GetKey(KeyCode.S) == true) walk(-1f, 0f);
        else if (Input.GetKey(KeyCode.A) == true) walk(1f, -1f);
        else if (Input.GetKey(KeyCode.D) == true) walk(1f, 1f);
        else if (Input.GetKey(KeyCode.Q)) agent.transform.Rotate(0, -120f * VECATime.fixedDeltaTime, 0);
        else if (Input.GetKey(KeyCode.E)) agent.transform.Rotate(0, 120f * VECATime.fixedDeltaTime, 0);        else walk(0f, 0f);

        if (kicking)
        {
            Debug.Log("kicking : " + kicking + " " + nextFrame);
            nextFrame += VECATime.fixedDeltaTime;
            //anim.PlayInFixedTime("Kick Up", 0, nextFrame);
            if (!anim.GetCurrentAnimatorStateInfo(0).IsName("Kick Up") || nextFrame > 1f)
            {
                kicking = false;
                nextFrame = 0f;
            }
        }

        if (ball != null)
        {
            //if (ballIsKickable() && anim.GetCurrentAnimatorStateInfo(0).IsName("Kick Up"))//&& onceFlag)
            if (ballIsKickable())
            {
                ballKicking = true;
                work("kick");
                //onceFlag = false;
            }
            if (ballKicking)
            {
                kickTheBall();
                ballKicking = false;
                reward = 1.0F;
                T = 10;
            }
        }
        if (T > 0) T -= 1;    }

    public bool isKickable()    {        if (ball != null) return ballIsKickable();        return false;    }

    bool ballIsKickable()
    {
        if (T > 0) return false;
        ballDistance = Vector3.Distance(agent.transform.position + 0.5f * agent.transform.TransformDirection(new Vector3(0, 0, 1)), ball.transform.position);
        if ((ball != null) && (ballDistance < limit) && (ball.transform.position.y < 2f))
        {
            Debug.Log("Ball is kickable!");
            return true;
        }
        return false;
    }

    private void kickTheBall()    {        Vector3 shootingDirection = (ball.transform.position - agent.transform.position).normalized;        ball.transform.position = (anim.GetBoneTransform(HumanBodyBones.LeftHand).position            + anim.GetBoneTransform(HumanBodyBones.RightHand).position) / 2            + agent.transform.forward * 1.5f;        shootingDirection += new Vector3(UnityEngine.Random.Range(-1f, 1f), 0, UnityEngine.Random.Range(-1f, 1f));        int power = 100 * UnityEngine.Random.Range(15, 30);        ball.GetComponent<Rigidbody>().AddForce(shootingDirection * power);    }

    public void setBall(GameObject newBall)
    {
        ball = newBall;
        freeBall();
    }

    public void freeBall()
    {
        ballKicking = false;
    }    private void OnCollisionEnter(Collision collision)    {        Debug.Log("collision with " + collision.other.name);    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Wall") || other.CompareTag("Object"))
        {
            triggered = true;
            isCollide = true;
        }
    }

    private void OnTriggerStay(Collider other)
    {
        if (other.CompareTag("Wall") || other.CompareTag("Object"))
        {
            triggered = true;
            isCollide = true;
            keepDistance(other.gameObject);
        }
    }

    private void OnTriggerExit(Collider other)
    {
        triggered = false;    }    private void keepDistance(GameObject obj)    {        Vector3 nearPoint = obj.GetComponent<Collider>().ClosestPointOnBounds(agent.transform.position);        Vector3 temp = (agent.transform.position - nearPoint).normalized * distance + nearPoint;        temp.y = 0;        agent.transform.position = temp;    }

    public Vector3 getRelativePos()
    {
        Vector3 dr = ball.transform.position - head.transform.position;
        Vector3 RelativePos = head.transform.InverseTransformDirection(dr);
        // Debug.Log(RelativePos);
        return RelativePos;
    }
}

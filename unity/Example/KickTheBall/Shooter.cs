using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Shooter : MonoBehaviour
{
    public GameObject ball;
    public GameObject baby;
    public GameObject babyHQ;
    public GameObject[] shooters;

    GameObject theShooter;
    GameObject clone;
    private bool noAudio;
    float speed;
    

    // Start is called before the first frame update
    void Start()
    {
        //shooters = GameObject.FindGameObjectsWithTag("Shooter");
        //baby = GameObject.Find("Baby");
    }

    public void SetAudio(bool _noAudio)
    {
        noAudio = _noAudio;
    }

    public void StartShooting()
	{
        Shoot();
	}

    private void Shoot()
    {
        if (clone != null) Destroy(clone);
        theShooter = shooters[Random.Range(0,shooters.Length)];

        speed = Random.Range(300, 500);
        Vector3 shootingDirection = (theShooter.transform.position - baby.transform.position).normalized;
        shootingDirection += new Vector3(Random.Range(-1f, 1f), 0, Random.Range(-1f, 1f));

        Vector3 shootingPoint = Vector3.Lerp(theShooter.transform.position, baby.transform.position, 0.01f);
        clone = Instantiate(ball, shootingPoint, theShooter.transform.rotation);
        clone.transform.localScale = new Vector3(2, 2, 2);
        clone.GetComponent<Rigidbody>().AddForce(shootingDirection * speed);
        babyHQ.GetComponent<KickTheBallInteract>().setBall(clone);
        clone.GetComponent<AudioSource>().mute = noAudio;
        clone.transform.parent = gameObject.transform;
    }
}

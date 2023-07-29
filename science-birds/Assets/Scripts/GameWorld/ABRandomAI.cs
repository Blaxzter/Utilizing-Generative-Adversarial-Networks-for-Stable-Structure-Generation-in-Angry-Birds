using System;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

namespace GameWorld
{
    public class ABRandomAI
    {
        private ABGameWorld _gameWorld;
        
        private GameObject _slingshot;
        private List<ABPig> _pigs;
        private List<ABBird> _birds;

        private Vector3? _prevTarget = null;
        
        private static float[] _launchVelocity = {2.9f,   2.88f,  2.866f, 2.838f, 2.810f, 2.800f, 2.790f, 2.773f, 2.763f, 2.745f, 2.74f, 2.735f, 2.73f};
        private static float[] _launchAngle = {0.13f,  0.215f, 0.296f, 0.381f, 0.476f, 0.567f, 0.657f, 0.741f, 0.832f, 0.924f, 1.014f, 1.106f, 1.197f};
        private static float[] _changeAngle = {0.052f, 0.057f, 0.063f, 0.066f, 0.056f, 0.054f, 0.050f, 0.053f, 0.042f, 0.038f, 0.034f, 0.029f, 0.025f};
        
        private static float BOUND = 0.1f;
        
        public ABRandomAI(ABGameWorld gameWorld, GameObject slingshot, List<ABPig> pigs, List<ABBird> birds)
        {
            _gameWorld = gameWorld;
            _slingshot = slingshot;
            _pigs = pigs;
            _birds = birds;
        }

        public bool Solve()
        {
            Debug.Log("Start AI Agent");
            // while (_pigs.Count != 0)
            // {
            
                // java.awt.Rectangle[x=160,y=260,width=26,height=96]
                // java.awt.Point[x=668,y=292]
            
                ABPig currentPig = _pigs[Random.Range(0, _pigs.Count)];
                Vector3 target = currentPig.GetBounds().center;
                
                if (_prevTarget != null && Vector3.Distance(_prevTarget.Value, target) < 10) {
                    float angle = Random.value * Mathf.PI * 2;
                    target.x = target.x + (int) (Mathf.Cos(angle) * 10);
                    target.y = target.y + (int) (Mathf.Sin(angle) * 10);
                    Debug.Log("Randomly changing to " + target);
                }
                
                _prevTarget = new Vector3(target.x, target.y, 0);

                float x = target.x;
                float y = target.y;
                
                
                // first estimate launch angle using the projectile equation (constant velocity)
                float v = _launchVelocity[6];
                float v2 = v * v;
                float v4 = v2 * v2;
                float tangent1 = (v2 - Mathf.Sqrt(v4 - (x * x + 2 * y * v2))) / x;
                float tangent2 = (v2 + Mathf.Sqrt(v4 - (x * x + 2 * y * v2))) / x;
                float t1 = Mathf.Atan(tangent1);
                float t2 = Mathf.Atan(tangent2);

                float bestError = 1000;
                float theta1 = 0;
                float theta2 = 0;
                
                List<Vector3> pts = new List<Vector3>();

                // search angles in range [t1 - BOUND, t1 + BOUND]
                for (float theta = t1 - BOUND; theta <= t1 + BOUND; theta += 0.001f)
                {
                    float velocity = getVelocity(theta);
                    
                    // initial velocities
                    float u_x = velocity * Mathf.Cos(theta);
                    float u_y = velocity * Mathf.Sin(theta);
                    
                    // the normalised coefficients
                    float a = -0.5f / (u_x * u_x);
                    float b = u_y / u_x;
                    
                    // the error in y-coordinate
                    float error = Mathf.Abs(a*x*x + b*x - y);
                    if (error < bestError)
                    {
                        theta1 = theta;
                        bestError = error;
                    }
                }
                if (bestError < 1000)
                {
                    theta1 = actualToLaunch(theta1);
                    // add launch points to the list
                    pts.Add(findReleasePoint(theta1));
                }
                bestError = 1000;
                
                // search angles in range [t2 - BOUND, t2 + BOUND]
                for (float theta = t2 - BOUND; theta <= t2 + BOUND; theta += 0.001f)
                {
                    float velocity = getVelocity(theta);
                    
                    // initial velocities
                    float u_x = velocity * Mathf.Cos(theta);
                    float u_y = velocity * Mathf.Sin(theta);
                    
                    // the normalised coefficients
                    float a = -0.5f / (u_x * u_x);
                    float b = u_y / u_x;
                    
                    // the error in y-coordinate
                    float error = Mathf.Abs(a*x*x + b*x - y);
                    if (error < bestError)
                    {
                        theta2 = theta;
                        bestError = error;
                    }
                }
                
                theta2 = actualToLaunch(theta2);
                
                //Debug.Log("Two angles: " + Mathf.toDegrees(theta1) + ", " + Mathf.toDegrees(theta2));
                    
                
                // add the higher point if it is below 75 degrees and not same as first
                if (theta2 < Mathf.Deg2Rad * 75f && Math.Abs(theta2 - theta1) > 0.000001f && bestError < 1000)
                    pts.Add(findReleasePoint(theta2));
            
                Vector3 releasePoint = new Vector3();
                
                if (pts.Count > 1) 
                {
                    releasePoint = pts[1];
                }
                else if (pts.Count == 1)
                    releasePoint = pts[0];
                else if (pts.Count == 2)
                {
                    // randomly choose between the trajectories, with a 1 in
                    // 6 chance of choosing the high one
                    int xcount = Random.Range(0, 4);
                    if (xcount == 0)
                        releasePoint = pts[1];
                    else
                        releasePoint = pts[0];
                }
                else
                if(pts.Count == 0)
                {
                    Debug.Log("No release point found for the target");
                    Debug.Log("Try a shot with 45 degree");
                    releasePoint = findReleasePoint(Mathf.PI/4f);
                }
                
                //Calculate the tapping time according the bird type 
                if (releasePoint != null) {
                    double releaseAngle = getReleaseAngle(releasePoint);
                    Debug.Log("Release Point: " + releasePoint);
                    Debug.Log("Release Angle: " + Mathf.Rad2Deg * releaseAngle);
                    int tapInterval = 0;

                    ABBird currentBird = _gameWorld.GetCurrentBird();
                    
                    if (currentBird is ABBirdYellow)
                    {
                        int range = Random.Range(0, 25);
                        tapInterval = 65 + range; // 65-90% of the way
                    }
                    else if (currentBird is ABBirdWhite) {
                        int range = Random.Range(0, 20);
                        tapInterval =  70 + range; // 70-90% of the way
                    }
                    else if (currentBird is ABBirdBlack) {
                        int range = Random.Range(0, 20);
                        tapInterval =  70 + range; // 70-90% of the way
                    }
                    else if (currentBird is ABBBirdBlue) {
                        int range = Random.Range(0, 20);
                        tapInterval =  65 + range; // 65-85% of the way
                    }
                    else
                        tapInterval =  0;

                    Vector3 slingSpot = getReferencePoint();
                    
                    float dx = releasePoint.x - slingSpot.x;
                    float dy = releasePoint.y - slingSpot.y;

                    
                    float dragX = slingSpot.x;
                    float dragY = slingSpot.y;

                    float dragDX = dragX + dx;
                    float dragDY = dragY + dy;

                    Vector2 deltaPos = new Vector2 (dragDX, dragDY);

                    Debug.Log ("POS = " + slingSpot);
                    Debug.Log ("DRAG = " + deltaPos);
                    
                    currentBird.DragBird(deltaPos * 10);
                    currentBird.LaunchBird();
                }
            // }

            
            return true;
        }
        
        
        private float actualToLaunch(float theta)
        {
            for (int i = 1; i < _launchAngle.Length; i++)
            {
                if (theta > _launchAngle[i-1] && theta < _launchAngle[i])
                    return theta + _changeAngle[i-1];
            }
            return theta + _changeAngle[_launchAngle.Length-1];
        }

        private float getVelocity(float theta)
        {
            if (theta < _launchAngle[0])    
                return 1.005f * _launchVelocity[0];
        
            for (int i = 1; i < _launchAngle.Length; i++)
            {
                if (theta < _launchAngle[i])
                    return 1.005f * _launchVelocity[i-1];
            }
        
            return 1.005f * _launchVelocity[_launchVelocity.Length-1];
        }
        
        public Vector3 findReleasePoint(float theta)
        {
            Vector3 releasePoint = _slingshot.gameObject.GetComponent<Collider2D>().bounds.center;
            Vector3 release = new Vector3((int)(releasePoint.x - 10 * Mathf.Cos(theta)), (int)(releasePoint.y + 10 * Mathf.Sin(theta)));
            return release;
        }
        
        public Vector3 getReferencePoint()
        {
            return _slingshot.gameObject.GetComponent<Collider2D>().bounds.center;
        }
        
        public double getReleaseAngle(Vector3 releasePoint)
        {
            Vector3 refPoint = getReferencePoint();

            return -Mathf.Atan2(refPoint.y - releasePoint.y, refPoint.x - releasePoint.x);
        }
    }
}
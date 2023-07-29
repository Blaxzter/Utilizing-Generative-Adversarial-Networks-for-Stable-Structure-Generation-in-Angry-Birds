// SCIENCE BIRDS: A clone version of the Angry Birds game used for 
// research purposes
// 
// Copyright (C) 2016 - Lucas N. Ferreira - lucasnfe@gmail.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>
//

using System;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Random = UnityEngine.Random;

public class ABBird : ABCharacter {

	public float _dragSpeed     = 1.0f;
    public float _dragRadius    = 1.0f;
    public float _launchGravity = 1.0f;

	public float _woodDamage  = 1.0f;
	public float _stoneDamage = 1.0f;
	public float _iceDamage   = 1.0f;

    public float _jumpForce   = 1.0f;
	public float _launchForce = 1.0f;

	public float _jumpTimer;
    public float _maxTimeToJump;

	public bool IsSelected      { get; set; }
	public bool IsFlying        { get; set; }
	public bool OutOfSlingShot  { get; set; }
	public bool JumpToSlingshot { get; set; }

	protected ABParticleSystem _trailParticles;

	protected override void Start ()
    {
		base.Start ();

        float nextJumpDelay = Random.Range(0.0f, _maxTimeToJump);
        Invoke("IdleJump", nextJumpDelay + 1.0f);

		_trailParticles = gameObject.AddComponent<ABParticleSystem> ();
		_trailParticles._particleSprites = ABWorldAssets.TRAIL_PARTICLES;
		_trailParticles._shootingRate = 0.1f;
    }

    void IdleJump()
    {
		if (IsFlying || OutOfSlingShot)
			return;

        if(IsIdle() && _rigidBody.gravityScale > 0f) {

			_rigidBody.AddForce(Vector2.up * _jumpForce);

			if (Random.value < 0.5f) {

				int randomSfx = Random.Range ((int)OBJECTS_SFX.MISC1, (int)OBJECTS_SFX.MISC2 + 1);
				// _audioSource.PlayOneShot (_clips [randomSfx]);
			}
		}

        float nextJumpDelay = Random.Range(0.0f, _maxTimeToJump);
        Invoke("IdleJump", nextJumpDelay + 1.0f);
    }

	private void CheckVelocityToDie() {

		if (_rigidBody.velocity.magnitude < 0.001f) {

			CancelInvoke ();
			Die ();
		}
	}

	// Used to move the camera towards the blocks only when bird is thrown to frontwards
	public bool IsInFrontOfSlingshot()
	{
		float slingXPos = ABGameWorld.Instance.Slingshot ().transform.position.x - ABConstants.SLING_SELECT_POS.x;
		return transform.position.x + _collider.bounds.size.x > slingXPos + _dragRadius * 2f;
	}

	public override void Die(bool withEffect = true)
	{
		ABGameWorld.Instance.KillBird (this);
		base.Die (withEffect);
	}

    public override void OnCollisionEnter2D(Collision2D collision)
    {
		if(OutOfSlingShot && !IsDying)
        {
			IsFlying = false;
			_trailParticles._shootParticles = false;

			ABGameWorld.Instance.RemoveLastTrajectoryParticle ();

			foreach (ABParticle part in _trailParticles.GetUsedParticles())
				ABGameWorld.Instance.AddTrajectoryParticle (part);

			InvokeRepeating("CheckVelocityToDie", 3f, 1f);
			_animator.Play("die", 0, 0f);

			IsDying = true;
        }
    }

    void OnTriggerEnter2D(Collider2D collider)
    {
		// Bird got dragged
        if(collider.tag == "Slingshot")
        {
            if(JumpToSlingshot)
				ABGameWorld.Instance.SetSlingshotBaseActive(false);

			// if(IsSelected && IsFlying)
			// 	_audioSource.PlayOneShot(_clips[(int)OBJECTS_SFX.DRAGED]);
        }
    }

    void OnTriggerStay2D(Collider2D collider)
    {
        if(collider.tag == "Slingshot")
        {
            if(JumpToSlingshot)
				ABGameWorld.Instance.SetSlingshotBaseActive(false);

            if(IsFlying)
            {
				OutOfSlingShot = true;

				Vector3 slingBasePos = ABGameWorld.Instance.Slingshot().transform.position - ABConstants.SLING_SELECT_POS;
                slingBasePos.z = transform.position.z + 0.5f;

				ABGameWorld.Instance.ChangeSlingshotBasePosition(slingBasePos);
				ABGameWorld.Instance.ChangeSlingshotBaseRotation (Quaternion.identity);
            }
        }
    }

	void OnTriggerExit2D(Collider2D collider)
	{
		if(collider.tag == "Slingshot")
		{
			// if(IsSelected && !IsFlying)
			// 	_audioSource.PlayOneShot(_clips[(int)OBJECTS_SFX.DRAGED]);
			//
			// if(!IsSelected && IsFlying)
			// 	_audioSource.PlayOneShot(_clips[(int)OBJECTS_SFX.FLYING]);
		}
	}
	
    public void SelectBird()
    {
		if (IsFlying || IsDying)
			return;
		
		IsSelected = true;

		// _audioSource.PlayOneShot (_clips[(int)OBJECTS_SFX.MISC1]);
        _animator.Play("selected", 0, 0f);

		ABGameWorld.Instance.SetSlingshotBaseActive(true);
    }

    public void SetBirdOnSlingshot()
    {
		Vector3 slingshotPos = ABGameWorld.Instance.Slingshot ().transform.position - ABConstants.SLING_SELECT_POS;
		transform.position = Vector3.MoveTowards(transform.position, slingshotPos, _dragSpeed * Time.deltaTime);

		if(Vector3.Distance(transform.position, slingshotPos) <= 0f)
		{
			JumpToSlingshot = false;
			OutOfSlingShot = false;
			_rigidBody.velocity = Vector2.zero;
		}
    }

	public void DragBird(Vector3 dragPosition)
	{		
		if (float.IsNaN(dragPosition.x) || float.IsNaN(dragPosition.y))
			return;
			
		dragPosition.z = transform.position.z;
		Vector3 slingshotPos = ABGameWorld.Instance.Slingshot ().transform.position - ABConstants.SLING_SELECT_POS;
		float deltaPosFromSlingshot = Vector2.Distance(dragPosition, slingshotPos);

        // Lock bird movement inside a circle
        if(deltaPosFromSlingshot > _dragRadius)
			dragPosition = (dragPosition - slingshotPos).normalized * _dragRadius + slingshotPos;

		transform.position = Vector3.Lerp(transform.position, dragPosition, _dragSpeed * Time.deltaTime);
		
		// Slingshot base look to slingshot
		Vector3 dist = ABGameWorld.Instance.DragDistance();
        float angle = Mathf.Atan2(dist.y, dist.x) * Mathf.Rad2Deg;
		ABGameWorld.Instance.ChangeSlingshotBaseRotation(Quaternion.AngleAxis(angle, Vector3.forward));

        // Slingshot base rotate around the selected point
		Collider2D col = _collider;
		ABGameWorld.Instance.ChangeSlingshotBasePosition ((transform.position - slingshotPos).normalized 
			* col.bounds.size.x / 2.25f + transform.position);
	}

	public void LaunchBird()
	{
		Vector3 slingshotPos = ABGameWorld.Instance.Slingshot ().transform.position - ABConstants.SLING_SELECT_POS;
		Vector2 deltaPosFromSlingshot = (transform.position - slingshotPos);
		_animator.Play("flying", 0, 0f);

		IsFlying = true;
		IsSelected = false;

		// The bird starts with no gravity, so we must set it
		_rigidBody.velocity = Vector2.zero;
		_rigidBody.gravityScale = _launchGravity;

		Vector2 f = -deltaPosFromSlingshot * _launchForce;

		_rigidBody.AddForce(f, ForceMode2D.Impulse);

		if(!ABGameWorld.Instance._isSimulation)
			_trailParticles._shootParticles = true;

		// _audioSource.PlayOneShot(_clips[(int)OBJECTS_SFX.SHOT]);
	}
}

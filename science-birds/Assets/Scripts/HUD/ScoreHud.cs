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

using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;

public class ScoreHud : ABSingleton<ScoreHud> {

    private ABParticleSystem  _scoreEmitter;

    // Use this for initialization
    void Start () {
	
        _scoreEmitter = GetComponent<ABParticleSystem> ();
        _scoreEmitter.SetParticlesParent (transform);
    }

    public void SpawnScorePoint(uint point, Vector3 position) {

        ABParticle scoreParticle = _scoreEmitter.ShootParticle ();
        if (!scoreParticle)
            return;

        scoreParticle.transform.rotation = Quaternion.identity;
        scoreParticle.transform.position = position;

        Text pointText = scoreParticle.GetComponent<Text>();
        pointText.text = point.ToString();

        HUD.Instance.AddScore (point);
    }
}
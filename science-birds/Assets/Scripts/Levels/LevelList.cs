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
 using System.Linq;
 using Levels;

public class LevelList : ABSingleton<LevelList> {

	private ABLevel[]   _levels;
	private Dictionary<int, LevelData> levelData = new Dictionary<int, LevelData>();

	public int CurrentIndex;
	private int _amountOfLevelsRequired;

	public int startIndex = 0;
	public int endIndex = -1;
	
	public ABLevel GetCurrentLevel() { 

		if (_levels == null)
			return null;

		if(CurrentIndex > _levels.Length - 1)
			return null;

		return _levels [CurrentIndex]; 
	}

	public void LoadLevelsFromSource(string[] levelSource, bool shuffle = false) {

		CurrentIndex = 0;

		_levels = new ABLevel[levelSource.Length];

		if(shuffle)
			ABArrayUtils.Shuffle(levelSource);

		for(int i = 0; i < levelSource.Length; i++)
			_levels[i] = LevelLoader.LoadXmlLevel(levelSource[i]);

		if (endIndex == -1)
			endIndex = _levels.Length;
	}

	// Use this for initialization
	public ABLevel NextLevel() {

		if(CurrentIndex == _levels.Length - 1)
			return null;

		ABLevel level = _levels [CurrentIndex];
		CurrentIndex++;

		return level;
	}

	// Use this for initialization
	public ABLevel SetLevel(int index) {

		if(index < 0 || index >= _levels.Length)
			return null;

		if (!levelData.ContainsKey(index))
		{
			Debug.Log("Create new Level Data: " + index);
			levelData.Add(index, new LevelData(index));
		}
		
		CurrentIndex = index;
		ABLevel level = _levels [CurrentIndex];

		return level;
	}

	public void ClearLevelData()
	{
		this.levelData.Clear();
	}

	public LevelData GetLevelData(int levelIndex)
	{
		if (!levelData.ContainsKey(levelIndex))
		{
			return null;
		}
		
		return levelData[levelIndex];
	}

	public LevelData GetCurrentLevelData()
	{
		return GetLevelData(CurrentIndex);
	}

	public bool AllLevelPlayed() {
		if (levelData.Count >= this._amountOfLevelsRequired)
		{
			return levelData.Values
				.Where(data => startIndex <= data.LevelIndex && data.LevelIndex <= endIndex)
				.Aggregate(true, (b, data) => b && data.HasBeenPlayed);
		}
		
		
		return false;
	}

	public Dictionary<int, LevelData> GetAllLevelData()
	{
		return levelData;
	}
	
	public void RequiredLevel(int amountOfLevelsRequired)
	{
		this._amountOfLevelsRequired = amountOfLevelsRequired;
	}

	public int AmountOfLoadedLevels()
	{
		return _levels.Length;
	}
}

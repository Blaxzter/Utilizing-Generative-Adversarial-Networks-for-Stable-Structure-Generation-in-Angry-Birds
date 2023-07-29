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

﻿using UnityEngine;
using UnityEditor;
using System;
using System.Reflection;
using System.Collections;
using System.Collections.Generic;
using UnityEditor.SceneManagement;

class LevelEditor : EditorWindow {
 
	public static BIRDS     _birdsOps;
	public static PIGS      _pigsOps;
	public static BLOCKS    _blocksOps;
	public static MATERIALS _material;

	private static int _birdsAdded = 0;
	private static int _birdsAmounInARow = 5;
	private static Vector3 _groundPos;

	[MenuItem ("Window/Level Editor %l")]
	static void Init () {

		if (EditorSceneManager.GetActiveScene ().name != "GameWorld") {

			if (EditorUtility.DisplayDialog ("Warning", "This window can only be opened in the Game World scene. " +
			   "Do you wanna go to that scene?", "Yes", "Cancel")) {

				EditorSceneManager.OpenScene ("Assets/Scenes/GameWorld.unity");
			}
			else 
				return;
		}
			
		LevelEditor window = (LevelEditor)EditorWindow.GetWindow(typeof(LevelEditor));
		window.Show ();

	}

	void OnEnable ()
	{
		_groundPos = new Vector3 (0f, -2.74f, 0f);
		_birdsAdded = GameObject.Find ("Birds").transform.childCount;
	}
		
	void OnFocus() {

		ToggleGizmos (false);

		HideLayer (LayerMask.NameToLayer("UI"));
	}

	void OnHierarchyChange() {

		if (EditorSceneManager.GetActiveScene ().name != "GameWorld")
			Close ();
	}

	void OnGUI()
	{
		EditorGUILayout.BeginHorizontal ();

		_birdsOps = (BIRDS) EditorGUILayout.EnumPopup("", _birdsOps);
		if (GUILayout.Button ("Create Bird", GUILayout.Width (80), GUILayout.Height (20))) {

			CreateBird (_birdsOps.ToString());
		}

		EditorGUILayout.EndVertical ();

		// Pigs section
		EditorGUILayout.BeginHorizontal ();

		_pigsOps = (PIGS) EditorGUILayout.EnumPopup("", _pigsOps);

		if (GUILayout.Button ("Create Pig", GUILayout.Width (80), GUILayout.Height (20))) {

			GameObject pig = InstantiateGameObject (ABWorldAssets.PIGS[_pigsOps.ToString()]);
			pig.transform.parent = GameObject.Find ("Blocks").transform;
		}

		EditorGUILayout.EndVertical ();

		// Blocks section
		EditorGUILayout.BeginHorizontal ();

		_blocksOps = (BLOCKS) EditorGUILayout.EnumPopup("", _blocksOps);
		_material = (MATERIALS) EditorGUILayout.EnumPopup("", _material);

		if (GUILayout.Button ("Create Block", GUILayout.Width (80), GUILayout.Height (20))) {

			GameObject block = InstantiateGameObject (ABWorldAssets.BLOCKS[_blocksOps.ToString()]);
			block.transform.parent = GameObject.Find ("Blocks").transform;

			BlockEditor.UpdateBlockMaterial (block.GetComponent<ABBlock>(), _material);
		}

		EditorGUILayout.EndHorizontal ();

		if (GUILayout.Button ("Create Platform")) {

			GameObject platform = InstantiateGameObject (ABWorldAssets.PLATFORM);
			platform.transform.parent = GameObject.Find ("Platforms").transform;
		}

		EditorGUILayout.BeginHorizontal ();

		if (GUILayout.Button ("Load Level"))
			LoadLevel ();

		if (GUILayout.Button ("Save Level"))
			SaveLevel ();

		if (GUILayout.Button ("Delete Level")) {

			if (EditorUtility.DisplayDialog ("Delete Level", 
				"Are you sure you want to delete this level?", "Yes", "Cancel")) {

				CleanLevel ();
			}
		}

		if (GUILayout.Button ("New Level")) {

			if (EditorUtility.DisplayDialog ("Create New Level",
				   "Are you sure you want to create a new level? This operation will " +
				"not save the current level in the scene", "Yes", "Cancel")) {

				CleanLevel ();

				Vector3 landscapePos = ABWorldAssets.LANDSCAPE.transform.position;
				GameObject landscape = InstantiateGameObject(ABWorldAssets.LANDSCAPE);
				landscape.transform.position = landscapePos;
				landscape.transform.parent = GameObject.Find ("GameWorld").transform;

				Vector3 backgroundPos = ABWorldAssets.BACKGROUND.transform.position;
				GameObject background = InstantiateGameObject(ABWorldAssets.BACKGROUND);
				background.transform.position = backgroundPos;
				background.transform.parent = GameObject.Find ("Camera").transform;

				GameObject slingshot = InstantiateGameObject(ABWorldAssets.SLINGSHOT);
				slingshot.transform.position = ABConstants.SLING_STANDARD_POS;
				slingshot.transform.parent = GameObject.Find ("GameWorld").transform;

				GameObject.Find ("Camera").GetComponent<ABGameplayCamera> ()._maxWidth = ABConstants.CAMERA_STANDARD_MAX_WIDTH;
				GameObject.Find ("Camera").GetComponent<ABGameplayCamera> ()._minWidth = ABConstants.CAMERA_STANDARD_MIN_WIDTH;
			}
		}
		


		EditorGUILayout.EndHorizontal ();
	}
		
	GameObject InstantiateGameObject(GameObject[]source, int index) {

		GameObject cube = (GameObject)PrefabUtility.InstantiatePrefab (source[index]);
		cube.transform.position = Vector3.zero;

		return cube;
	}

	GameObject InstantiateGameObject(GameObject source) {

		GameObject cube = (GameObject)PrefabUtility.InstantiatePrefab (source);
		cube.transform.position = Vector3.zero;

		return cube;
	}

	void CleanLevel() {

		_birdsAdded = 0;

		DestroyImmediate (GameObject.Find ("Birds").gameObject);
		GameObject birds = new GameObject ();
		birds.name = "Birds";
		birds.transform.parent = GameObject.Find ("GameWorld").transform;

		DestroyImmediate (GameObject.Find ("Blocks").gameObject);
		GameObject blocks = new GameObject ();
		blocks.name = "Blocks";
		blocks.transform.parent = GameObject.Find ("GameWorld").transform;

		DestroyImmediate (GameObject.Find ("Platforms").gameObject);
		GameObject plats = new GameObject ();
		plats.name = "Platforms";
		plats.transform.parent = GameObject.Find ("GameWorld").transform;

		DestroyImmediate (GameObject.Find ("Landscape"));
		DestroyImmediate (GameObject.Find ("Slingshot"));
		DestroyImmediate (GameObject.Find ("Background"));
	}

	void LoadLevel() {

		string path = EditorUtility.OpenFilePanel("Select level to open", "Assets/Resources/Levels/", "");

		if (path != "") {

			CleanLevel ();

			Debug.Log (path);

//			TextAsset levelFile = (TextAsset)Resources.Load (finalPath);
			string levelText = LevelLoader.ReadXmlLevel (path);
			Debug.Log (levelText);
			ABLevel level = LevelLoader.LoadXmlLevel (levelText);

			DecodeLevel (level);
		}
	}

	void SaveLevel() {

		string path = EditorUtility.SaveFilePanel("Select location to save level", "Assets/Resources/Levels/", "level.xml", "xml");

		if (path != "") {

			LevelLoader.SaveXmlLevel (EncodeLevel (), path);
		}
	}

	void CreateBird(string type) {

		GameObject bird = InstantiateGameObject (ABWorldAssets.BIRDS[type]);
		bird.name = bird.name + "_" + _birdsAdded;
		bird.transform.parent = GameObject.Find ("Birds").transform;

		Vector3 birdsPos = ABConstants.SLING_STANDARD_POS - ABConstants.SLING_SELECT_POS;

		// From the second Bird on, they are added to the ground
		if(_birdsAdded >= 1)
		{
			birdsPos.y = _groundPos.y;

			for(int i = 0; i < _birdsAdded; i++) {

				if ((i + 1) % _birdsAmounInARow == 0) {

					float coin = (UnityEngine.Random.value < 0.5f ? 1f : -1);
					birdsPos.x = ABConstants.SLING_SELECT_POS.x + (UnityEngine.Random.value * 0.5f * coin);
				}

				birdsPos.x -=  bird.GetComponent<SpriteRenderer>().bounds.size.x * 1.75f;
			}
		}

		bird.transform.position = birdsPos;

		_birdsAdded++;
	}

	public ABLevel EncodeLevel() 
	{
		ABLevel level = new ABLevel();

		level.birds  = new List<BirdData>(); 
		level.pigs   = new List<OBjData>();
		level.blocks = new List<BlockData>();
		level.platforms = new List<PlatData>();

		foreach (Transform child in GameObject.Find ("Birds").transform) {

			string type = child.name;
			level.birds.Add (new BirdData (type));
		}

		foreach (Transform child in GameObject.Find ("Blocks").transform) {

			string type = child.name;
			float x = child.transform.position.x;
			float y = child.transform.position.y;
			float rotation = child.transform.rotation.eulerAngles.z;

			if (child.GetComponent<ABPig> () != null) {

				level.pigs.Add (new OBjData (type, rotation, x, y));
			} 
			else if (child.GetComponent<ABBlock> () != null) {
				
				string material = child.GetComponent<ABBlock> ()._material.ToString ();
				level.blocks.Add (new BlockData (type, rotation, x, y, material));
			}

		}

		foreach (Transform child in GameObject.Find ("Platforms").transform) {

			PlatData obj = new PlatData ();

			obj.type = child.name;
			obj.x = child.transform.position.x;
			obj.y = child.transform.position.y;
			obj.rotation = child.transform.rotation.eulerAngles.z;
			obj.scaleX = child.transform.localScale.x;
			obj.scaleY = child.transform.localScale.y;

			level.platforms.Add (obj);
		}
			
		return level;
	}

	public void DecodeLevel(ABLevel level) 
	{
		foreach (OBjData gameObj in level.pigs) {

			Vector2 pos = new Vector2 (gameObj.x, gameObj.y);
			Quaternion rotation = Quaternion.Euler (0, 0, gameObj.rotation);

			GameObject pig = InstantiateGameObject (ABWorldAssets.PIGS[gameObj.type]);
			pig.transform.parent = GameObject.Find ("Blocks").transform;
			pig.transform.position = pos;
			pig.transform.rotation = rotation;
		}

		_birdsAdded = 0;
		foreach(BirdData gameObj in level.birds){
			
			CreateBird (gameObj.type);
			_birdsAdded++;
		}

		foreach(BlockData gameObj in level.blocks) {

			Vector2 pos = new Vector2 (gameObj.x, gameObj.y);
			Quaternion rotation = Quaternion.Euler (0, 0, gameObj.rotation);

			GameObject block = InstantiateGameObject (ABWorldAssets.BLOCKS[gameObj.type]);
			block.transform.parent = GameObject.Find ("Blocks").transform;
			block.transform.position = pos;
			block.transform.rotation = rotation;

			MATERIALS material = (MATERIALS)Enum.Parse(typeof(MATERIALS), gameObj.material);
			BlockEditor.UpdateBlockMaterial(block.GetComponent<ABBlock>(), material);
		}

		foreach(OBjData gameObj in level.platforms)
		{
			Vector2 pos = new Vector2 (gameObj.x, gameObj.y);
			Quaternion rotation = Quaternion.Euler (0, 0, gameObj.rotation);

			GameObject platform = InstantiateGameObject (ABWorldAssets.PLATFORM);
			platform.transform.parent = GameObject.Find ("Platforms").transform;
			platform.transform.position = pos;
			platform.transform.rotation = rotation;
		}
	}

	public static void ToggleGizmos(bool gizmosOn) {

		int val = gizmosOn ? 1 : 0;

		Assembly asm = Assembly.GetAssembly(typeof(Editor));
		Type type = asm.GetType("UnityEditor.AnnotationUtility");

		if (type != null) {

			MethodInfo getAnnotations = type.GetMethod("GetAnnotations", BindingFlags.Static | BindingFlags.NonPublic);
			MethodInfo setGizmoEnabled = type.GetMethod("SetGizmoEnabled", BindingFlags.Static | BindingFlags.NonPublic);
			MethodInfo setIconEnabled = type.GetMethod("SetIconEnabled", BindingFlags.Static | BindingFlags.NonPublic);

			var annotations = getAnnotations.Invoke(null, null);

			foreach (object annotation in (IEnumerable)annotations) {

				Type annotationType = annotation.GetType();
				FieldInfo classIdField = annotationType.GetField("classID", BindingFlags.Public | BindingFlags.Instance);
				FieldInfo scriptClassField = annotationType.GetField("scriptClass", BindingFlags.Public | BindingFlags.Instance);

				if (classIdField != null && scriptClassField != null) {
					int classId = (int)classIdField.GetValue(annotation);

					string scriptClass = (string)scriptClassField.GetValue(annotation);
					setGizmoEnabled.Invoke(null, new object[] { classId, scriptClass, val });
					setIconEnabled.Invoke(null, new object[] { classId, scriptClass, val });
				}
			}
		}
	}

	static void HideLayer(int layerNumber)
	{
		LayerMask layerNumberBinary = 1 << layerNumber;
		LayerMask blocksLayer = 1 << LayerMask.NameToLayer ("Blocks");
		LayerMask birdsLayer = 1 << LayerMask.NameToLayer ("Birds");
		LayerMask pigsLayer = 1 << LayerMask.NameToLayer ("Pigs");
		LayerMask platLayer = 1 << LayerMask.NameToLayer ("Platforms");

		Tools.visibleLayers = ~layerNumberBinary; 
		Tools.lockedLayers = ~(blocksLayer | birdsLayer | pigsLayer | platLayer);

		SceneView.RepaintAll();
	}
}
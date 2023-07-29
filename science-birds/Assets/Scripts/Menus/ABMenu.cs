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
using UnityEngine.UI;

public class ABMenu : MonoBehaviour
{

    public Text args = null;
    
    private static string GetArg(string name)
    {
        var args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == name && args.Length > i + 1)
            {
                return args[i + 1];
            }
        }
        return null;
    }
    
    private void Awake()
    {
        if (args != null)
        {
            var generatorPort = GetArg("generatorPort");
            if (generatorPort != null)
                args.text = int.Parse(GetArg("generatorPort")) + "";
            else
            {
                var cargs = System.Environment.GetCommandLineArgs();
                var printString = ""; 
                for (int i = 0; i < cargs.Length; i++)
                {
                    printString += "\n" + i + " " + cargs[i];
                }
                
                args.text = printString;
            }
        }
    }

    public void LoadNextScene(string sceneName) {

        ABSceneManager.Instance.LoadScene(sceneName);
    }

    public void LoadNextScene(string sceneName, bool loadTransition, ABSceneManager.ActionBetweenScenes action) {

        ABSceneManager.Instance.LoadScene(sceneName, loadTransition, action);
    }
}
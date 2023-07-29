using UnityEngine;

namespace Levels
{
    public class LevelData
    {
        private readonly int _levelIndex;

        private int _death = -1;
        private int _woodBlockDestroyed = 0;
        private int _iceBlockDestroyed = 0;
        private int _stoneBlockDestroyed = 0;
        
        private int _birdsUsed = -1;
        private float _initialDamage = -1;
        private float _cumulativeDamage = -1;

        private bool _isStable = true;
        private bool _hasBeenPlayed = false;
        private bool _won = false;

        private float _score = -1;


        public LevelData(int levelIndex)
        {
            _levelIndex = levelIndex;
        }

        public string GetJson()
        {
            return 
                "{" +
                    "\"level_index\": " + _levelIndex + "," +
                    "\"cumulative_damage\": " + _cumulativeDamage + "," +
                    "\"initial_damage\": " + _initialDamage + "," +
                    "\"is_stable\": " + (_isStable ? "true" : "false") + "," +
                    "\"death\": " + _death + "," +
                    "\"woodBlockDestroyed\": " + _woodBlockDestroyed + "," +
                    "\"iceBlockDestroyed\": " + _iceBlockDestroyed + "," +
                    "\"stoneBlockDestroyed\": " + _stoneBlockDestroyed + "," +
                    "\"birds_used\": " + _birdsUsed + "," +
                    "\"has_been_played\": " + (_hasBeenPlayed ? "true" : "false") + "," +
                    "\"won\": " + (_won ? "true" : "false") + "," +
                    "\"score\": " + _score  +
                "}";
        }
        
        public float CumulativeDamage
        {
            get { return _cumulativeDamage; }
            set { _cumulativeDamage = value; }
        }

        public float InitialDamage
        {
            get { return _initialDamage; }
            set { _initialDamage = value; }
        }

        public bool IsStable
        {
            get { return _isStable; }
            set { _isStable = value; }
        }

        public float Score
        {
            get { return _score; }
            set { _score = value; }
        }

        public bool HasBeenPlayed
        {
            get { return _hasBeenPlayed; }
            set { _hasBeenPlayed = value; }
        }

        public bool Won
        {
            get { return _won; }
            set { _won = value; }
        }

        public int Death
        {
            get { return _death; }
            set { _death = value; }
        }

        public int BirdsUsed
        {
            get { return _birdsUsed; }
            set { _birdsUsed = value; }
        }

        public int LevelIndex
        {
            get { return _levelIndex; }
        }

        public int WoodBlockDestroyed
        {
            get { return _woodBlockDestroyed; }
            set { _woodBlockDestroyed = value; }
        }

        public int IceBlockDestroyed
        {
            get { return _iceBlockDestroyed; }
            set { _iceBlockDestroyed = value; }
        }

        public int StoneBlockDestroyed
        {
            get { return _stoneBlockDestroyed; }
            set { _stoneBlockDestroyed = value; }
        }
    }
}
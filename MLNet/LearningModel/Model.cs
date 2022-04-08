using System.Text;

namespace MLNet.LearningModel
{
    /// <summary>
    ///     Learning Model
    /// </summary>
    public abstract class Model
    {
        protected Model(string name)
        {
            Name = name;
        }

        public string Name { get; set; }


        public abstract void Save(string path);

        public abstract void Load(string path);

        public override string ToString()
        {
            var strBuild = new StringBuilder();
            strBuild.AppendLine("Name");
            return strBuild.ToString();
        }
    }
}
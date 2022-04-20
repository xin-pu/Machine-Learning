using System.Text;
using Numpy;

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


        internal abstract NDarray call(NDarray x);

        internal abstract NDarray sgd(NDarray x, NDarray y, double learning_rate);

        internal abstract void fit(NDarray x, NDarray y, double learning_rate);

        public NDarray Call(NDarray x)
        {
            var res = call(x);
            Log.print?.Invoke($"{Name} Call:\r\n{res}");
            return res;
        }

        public NDarray Predict(NDarray x)
        {
            var res = call(x);
            Log.print?.Invoke($"{Name} Predict:\r\n{res}");
            return res;
        }

        public void Fit(NDarray x, NDarray y, double learning_rate = 0.1)
        {
            fit(x, y, learning_rate);
        }


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
using System.Text;
using MLNet.Loss;
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

        public abstract LossBase? CostFunc { set; get; }

        public abstract NDarray? Resolve { set; get; }

        public bool Print { set; get; } = true;

        public NDarray Call(NDarray x)
        {
            var x_cvt = convert(x);
            var res = call(x_cvt);
            Log.print?.Invoke($"{Name} Call:\r\n{res}");
            return res;
        }

        public NDarray Predict(NDarray x)
        {
            var x_cvt = convert(x);
            var res = call(x_cvt);
            Log.print?.Invoke($"{Name} Predict:\r\n{res}");
            return res;
        }

        public void Fit(NDarray x, NDarray y, double learning_rate = 0.1, int epoch = 100)
        {
            try
            {
                Log.print?.Invoke($"{Name} Start Fit:\r\n");
                var x_cvt = convert(x);
                fit(x_cvt, y, learning_rate, epoch);
            }
            catch (Exception ex)
            {
                Log.print?.Invoke($"{Name} Predict:\r\n{ex.Message}");
            }
        }


        public virtual void Save(string path)
        {
        }

        public virtual void Load(string path)
        {
        }

        public override string ToString()
        {
            var strBuild = new StringBuilder();
            strBuild.AppendLine("Name");
            return strBuild.ToString();
        }

        #region internal

        internal abstract NDarray convert(NDarray x);

        internal abstract void fit(NDarray x, NDarray y, double learning_rate, int epoch);

        internal abstract NDarray call(NDarray x);

        #endregion

        #region serialize

        #endregion
    }
}
﻿using System.Text;
using AutoDiff;
using MLNet.LearningModel;
using MLNet.Loss;
using MLNet.Utils;
using Numpy;

namespace MLNet.Model
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

        public abstract LossBase CostFunc { set; get; }

        public abstract NDarray Resolve { set; get; }

        public bool Print { set; get; } = true;

        public NDarray Call(NDarray x)
        {
            var x_cvt = convert(x);
            var res = call(x_cvt);
            print($"{Name} Call:\r\n{res}");
            return res;
        }

        public NDarray Predict(NDarray x)
        {
            var x_cvt = convert(x);
            var res = call(x_cvt);

            print($"{Name} Predict:\r\n{res}");
            return res;
        }

        public Metric Evaluate(NDarray x, NDarray y)
        {
            var delta_abs = np.abs(Predict(x) - y);
            var mad = delta_abs.GetData<double>().Average();

            var delta_mse = np2.power(np.abs(Predict(x) - y), 2);
            var mse = delta_mse.GetData<double>().Average();

            return new Metric
            {
                MAD = mad,
                MSE = mse
            };
        }

        public void Fit(NDarray x, NDarray y, double learning_rate = 0.1, int epoch = 100)
        {
            try
            {
                print($"{Name} Start Fit:\r\n");

                /// Step 1 Convert Model
                var x_cvt = convert(x);

                /// Step 2 Create Loss Function
                var featureCount = x_cvt.shape[1];
                var w = Enumerable.Range(0, featureCount).Select(_ => new Variable()).ToArray();
                CostFunc = initialLoss(w, x_cvt, y);

                /// Step 3 Fit
                fit(x_cvt, y, learning_rate, epoch);
            }
            catch (Exception ex)
            {
                print($"{Name} Predict:\r\n{ex.Message}");
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
            strBuild.AppendLine($"{Name}");
            strBuild.AppendLine($"Resolve:\r\n{Resolve}");
            return strBuild.ToString();
        }


        #region print

        internal void print(object obj)
        {
            if (Print)
                Log.print?.Invoke(obj);
        }

        public void PrintSelf()
        {
            Log.print?.Invoke(this);
        }

        internal void printTitle(string title)
        {
        }

        #endregion

        #region internal

        internal abstract NDarray convert(NDarray x);

        internal abstract void fit(NDarray x, NDarray y, double learning_rate, int epoch);

        internal abstract NDarray call(NDarray x);

        internal abstract LossBase initialLoss(Variable[] variables, NDarray x, NDarray y);

        #endregion

        #region serialize

        #endregion
    }
}
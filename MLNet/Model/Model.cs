using System.Text;
using AutoDiff;
using MLNet.Kernels;
using MLNet.LearningModel;
using MLNet.Loss;
using MLNet.Utils;
using Numpy;
using YAXLib;
using YAXLib.Attributes;

namespace MLNet.Model
{
    /// <summary>
    ///     Learning Model
    /// </summary>
    public abstract class Model : ViewModelBase
    {
        protected Model(string name)
        {
            Name = name;
        }

        protected Model()
        {
        }


        public string? Name { get; set; }

        [YAXDontSerialize] public LossBase CostFunc { set; get; } = null!;

        [YAXDontSerialize] public NDarray Resolve { set; get; } = null!;

        [YAXDontSerialize] public Kernel? Kernel { set; get; }


        public string FilePath => $"{Name}.xml";

        public bool Print { set; get; } = true;

        public NDarray Call(NDarray x)
        {
            return Predict(x);
        }

        public NDarray Predict(NDarray x)
        {
            /// Step 1 Transform
            var x_cvt = transform(x);

            /// Step 2 Casll
            var res = call(x_cvt);


            return res;
        }

        public Metric Evaluate(NDarray x, NDarray y)
        {
            var y_pred = Predict(x);
            return new Metric(y, y_pred);
        }

        public void Fit(NDarray x, NDarray y, double learning_rate = 0.1, int epoch = 100, int batchsize = 8)
        {
            try
            {
                print($"{Name} Start Fit:\r\n");

                /// Step 1 Convert Model
                var x_cvt = transform(x);

                /// Step 2 Create Loss Function
                var featureCount = x_cvt.shape[1];
                var w = Enumerable.Range(0, featureCount).Select(_ => new Variable()).ToArray();
                CostFunc = initialLoss(w, x_cvt, y);

                /// Step 3 Fit
                fit(x_cvt, y, learning_rate, epoch, batchsize);

                /// Step 4 Evalate
                var metric = Evaluate(x, y);
                print(metric);
            }
            catch (Exception ex)
            {
                print($"{Name} Predict:\r\n{ex.Message}");
            }
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

        internal virtual NDarray transform(NDarray x)
        {
            if (Kernel == null)
                return transformer.to_linear_firstorder(x);

            var kernel = Kernel.Transform(x);
            var final = transformer.to_linear_firstorder(kernel);
            return final;
        }

        internal abstract void fit(NDarray x, NDarray y, double learning_rate, int epoch, int batchsize);

        internal abstract NDarray call(NDarray x);

        internal abstract LossBase initialLoss(Variable[] variables, NDarray x, NDarray y);

        #endregion

        #region serialize

        public virtual void Save(string path)
        {
            using var stream = File.Open(path, FileMode.Create, FileAccess.Write, FileShare.Read);
            using var textWriter = new StreamWriter(stream);
            var serializer = new YAXSerializer(typeof(Model));
            serializer.Serialize(this, textWriter);
            Resolve.tofile(FilePath, "", "");
            textWriter.Flush();
        }

        public static Model Load(string path)
        {
            using var stream = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var textReader = new StreamReader(stream);
            var type = typeof(Model);
            var deserializer = new YAXSerializer(type);
            var model = (Model) deserializer.Deserialize(textReader);
            var r = np.fromfile(model.FilePath);
            model.Resolve = np.expand_dims(r, -1);
            return model;
        }

        #endregion
    }
}